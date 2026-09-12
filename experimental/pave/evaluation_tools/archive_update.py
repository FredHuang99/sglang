"""Explicit, resumable publication transactions; existing raw runs are immutable."""
import os
import shutil
from pathlib import Path

from pave_sim.evaluation import campaign as c
from pave_sim.records import write_json
from .archive import list_files, refresh_checksums, verify_archive

CHECKSUMS = 'provenance/checksums.json'

def inventory(root):
    return {p.relative_to(root).as_posix(): {'sha256': c.file_sha(p), 'bytes': p.stat().st_size}
            for p in list_files(root)}

def immutable(relative):
    parts = Path(relative).parts
    return (relative.startswith('archive/campaign-001/') or
            (relative.startswith('archive/campaign-002/') and any(x in parts for x in ('runs', 'inputs'))) or
            relative.startswith(('archive/original_inputs/', 'archive/source_and_environment/src/')))

def prepare_update(staged, destination, update_id, allowed_changes, *, retire=()):
    """Build an exact allowlisted transaction and real historical copies in staging."""
    staged, destination = Path(staged), Path(destination)
    if not update_id.replace('_', '').isalnum():
        raise ValueError('Unsafe update ID')
    verify_archive(destination)
    verify_archive(staged)
    old, new = inventory(destination), inventory(staged)
    retired=set(retire)
    if retired != old.keys()-new.keys() or any(not p.startswith('paper_tables/f_low_') for p in retired):
        raise ValueError('Archive update cannot delete historical files')
    changed = sorted(p for p in old.keys() & new.keys() if old[p] != new[p])
    permitted = set(allowed_changes) | {CHECKSUMS}
    if any(immutable(p) or p not in permitted for p in changed):
        raise ValueError('Unapproved/immutable archive change: ' + repr([p for p in changed if immutable(p) or p not in permitted]))
    history = 'provenance/report_history/' + update_id
    if any(p.startswith(history + '/') for p in new):
        raise ValueError('Historical update already staged')
    for relative in changed+sorted(retired):
        target = staged / history / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(destination / relative, target)
        if c.file_sha(target) != old[relative]['sha256']:
            raise ValueError('Historical copy mismatch')
    plan = {'update_id': update_id, 'old': old, 'changed': changed,
            'retired':sorted(retired),
            'unchanged': sorted(p for p in old.keys() & new.keys() if old[p] == new[p]),
            'added_payload': sorted(new.keys() - old.keys()),
            'history_prefix': history, 'rule': 'old-or-exact-new on resume; no raw overwrite'}
    receipt = f'provenance/archive_receipts/{update_id}.json'
    write_json(staged / receipt, {'status': 'complete', 'update_id': update_id,
        'changed': changed, 'added_payload': plan['added_payload'], 'history_prefix': history,
        'meaning': 'Valid only with the matching full checksums inventory; written last before checksums.'})
    write_json(staged / f'provenance/archive_receipts/{update_id}_plan.json', plan)
    refresh_checksums(staged)
    plan['new'] = inventory(staged)
    plan['receipt'] = receipt
    return plan

def apply_update(staged, destination, plan, journal_path, *, stop_after=None):
    """Copy atomically, validating old identity before every replacement.

    stop_after is an I/O fault injection for synthetic publication tests only.
    Journal is outside the published tree, and retained on any interruption.
    """
    staged, destination, journal_path = Path(staged), Path(destination), Path(journal_path)
    if journal_path.resolve().is_relative_to(destination.resolve()):
        raise ValueError('Transaction journal must be outside the archive')
    verify_archive(staged)
    if inventory(staged) != plan['new']:
        raise ValueError('Staged transaction changed')
    if journal_path.exists():
        journal = c.read(journal_path)
        if journal['plan'] != plan:
            raise ValueError('Resume transaction identity mismatch')
    else:
        verify_archive(destination)
        if inventory(destination) != plan['old']:
            raise ValueError('Old archive changed before publication')
        journal = {'status': 'incomplete', 'plan': plan, 'applied': []}
        c.atomic_json(journal_path, journal)
    actual = inventory(destination)
    retired=set(plan.get('retired',[]))
    if not actual.keys() <= (plan['new'].keys() | retired) or not (plan['old'].keys()-retired) <= actual.keys():
        raise ValueError('Unexpected destination inventory during update')
    for relative, value in actual.items():
        if value != plan['old'].get(relative) and value != plan['new'].get(relative):
            raise ValueError('Old digest mismatch during update: ' + relative)
    needed = sum(v['bytes'] for p, v in plan['new'].items() if actual.get(p) != v)
    if shutil.disk_usage(destination).free < needed + 64 * 1024**2:
        raise ValueError('Insufficient update space')
    paths = sorted(p for p in plan['new'] if p not in (CHECKSUMS, plan['receipt'])) + [plan['receipt'], CHECKSUMS]
    done = 0
    for relative in paths:
        if relative==plan['receipt']:
            for obsolete in sorted(retired):
                target=destination/obsolete
                historical=destination/plan['history_prefix']/obsolete
                if not target.resolve().is_relative_to(destination.resolve()) or immutable(obsolete):
                    raise ValueError('Unsafe presentation retirement')
                if c.file_sha(historical)!=plan['old'][obsolete]['sha256']:
                    raise ValueError('Retirement historical copy mismatch')
                if target.exists():
                    if c.file_sha(target)!=plan['old'][obsolete]['sha256']:
                        raise ValueError('Retired presentation changed')
                    target.unlink()
        wanted = plan['new'][relative]
        target = destination / relative
        if not target.resolve().is_relative_to(destination.resolve()):
            raise ValueError('Unsafe publication path')
        current = {'sha256': c.file_sha(target), 'bytes': target.stat().st_size} if target.exists() else None
        if current == wanted:
            continue
        if current != plan['old'].get(relative):
            raise ValueError('Old digest mismatch before replace: ' + relative)
        if current is not None and immutable(relative):
            raise ValueError('Immutable raw file replacement')
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_name(target.name + '.publish.tmp')
        try:
            shutil.copyfile(staged / relative, temp)
            if c.file_sha(temp) != wanted['sha256'] or temp.stat().st_size != wanted['bytes']:
                raise ValueError('Staging copy changed')
            os.replace(temp, target)
        finally:
            if temp.exists():
                temp.unlink()
        journal['applied'].append(relative)
        c.atomic_json(journal_path, journal)
        done += 1
        if stop_after is not None and done == stop_after:
            raise RuntimeError('Synthetic publication interruption')
    result = verify_archive(destination)
    journal['status'] = 'complete'
    journal['verification'] = result
    c.atomic_json(journal_path, journal)
    return result
