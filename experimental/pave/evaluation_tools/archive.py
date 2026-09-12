"""Lossless, hash-verified archives; no model execution and no symlinks."""
import os
import shutil
from pathlib import Path

from pave_sim.evaluation import campaign as c
from pave_sim.records import write_json


def list_files(root):
    root = Path(root)
    is_junction = getattr(os.path, 'isjunction', lambda path: False)
    if not root.is_dir() or root.is_symlink() or is_junction(root):
        raise ValueError(f'Expected a real archive source directory: {root}')
    items = []
    for current, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in ('__pycache__', '.pytest_cache', '.venv') and not d.startswith('tmp-'))
        for name in dirs:
            p = Path(current) / name
            if p.is_symlink() or is_junction(p):
                raise ValueError(f'Archive source contains a directory link: {p}')
        for name in sorted(files):
            p = Path(current) / name
            if p.is_symlink():
                raise ValueError(f'Archive source is a symlink: {p}')
            if name == 'execution.lock' or name.endswith(('.pyc', '.tmp')):
                continue
            items.append(p)
    return items


def copy_checked(entries, destination):
    destination = Path(destination)
    paths = [relative for _, relative in entries]
    if len(paths) != len(set(paths)):
        raise ValueError('Duplicate archive destination')
    total = sum(p.stat().st_size for p, _ in entries)
    parent = destination
    while not parent.exists():
        parent = parent.parent
    if shutil.disk_usage(parent).free < total + 64 * 1024**2:
        raise ValueError('Insufficient archive space including 64 MiB reserve')
    checksums = {}
    # Fail on all detected existing-file conflicts before copying anything.
    for source, relative in entries:
        target = destination / relative
        if target.resolve() == destination.resolve() or not target.resolve().is_relative_to(destination.resolve()):
            raise ValueError('Unsafe archive relative path')
        checksum = c.file_sha(source)
        checksums[relative] = {'sha256': checksum, 'bytes': source.stat().st_size}
        if target.exists() and (not target.is_file() or c.file_sha(target) != checksum):
            raise ValueError(f'Archive conflict: {relative}')
    for source, relative in entries:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(source, target)
        if c.file_sha(target) != checksums[relative]['sha256'] or target.stat().st_size != checksums[relative]['bytes']:
            raise ValueError(f'Archive copy verification failed: {relative}')
    return checksums


def assemble(root, destination, evidence=None, *, source_dir=None, original_trace=None):
    """Archive one explicitly selected campaign without neighboring work copies.

    Campaign inputs already contain its deployment, profile and request lists.
    Optional evidence and original trace are included only when explicitly given.
    """
    root, destination = Path(root).resolve(), Path(destination).resolve()
    workspace = Path(source_dir).resolve() if source_dir else Path(__file__).resolve().parent.parent
    manifest, _ = c.load_campaign(root)
    if (root / 'execution.lock').exists():
        raise ValueError('Cannot archive a running campaign')
    if destination.exists():
        raise ValueError('Archive output must be a new directory')
    sources = [root, workspace] + ([Path(evidence).resolve()] if evidence else [])
    if any(destination.is_relative_to(p) for p in sources):
        raise ValueError('Archive output must be outside all source directories')
    # Never present current code as the source of a different runtime version.
    for name, checksum in manifest['source']['runtime_files'].items():
        if c.file_sha(workspace / 'src' / name) != checksum:
            raise ValueError(f'Runtime snapshot does not match campaign: {name}')
    entries = []
    def add_tree(source, prefix):
        for p in list_files(source):
            entries.append((p, prefix + '/' + p.relative_to(source).as_posix()))
    add_tree(root, 'archive/' + root.name)
    raw = Path(original_trace).resolve() if original_trace else None
    if raw:
        expected = manifest['configuration']['expected_raw_sha256']
        if expected and c.file_sha(raw) != expected:
            raise ValueError('Original trace changed')
        entries.append((raw, 'archive/original_inputs/' + raw.name))
    for name in ('src', 'evaluation_tools', 'tools', 'docs', 'examples'):
        if (workspace / name).is_dir():
            add_tree(workspace / name, 'archive/source_and_environment/' + name)
    for name in ('pyproject.toml', 'README.md', 'LICENSE', 'NOTICE'):
        entries.append((workspace / name, 'archive/source_and_environment/' + name))
    if evidence:
        add_tree(Path(evidence).resolve(), 'archive/evidence')
    copied = copy_checked(entries, destination)
    write_json(destination / 'provenance/archive_receipts/raw_archive.json',
               {'files':len(copied), 'bytes':sum(r['bytes'] for r in copied.values()),
                'campaign_runs':len(manifest['runs']), 'original_trace_sha256':c.file_sha(raw) if raw else None,
                'all_copied_files_verified':True})
    refresh_checksums(destination)
    return {'files':len(copied), 'bytes':sum(r['bytes'] for r in copied.values())}


def refresh_checksums(destination):
    destination = Path(destination)
    # Checksums exclude only their own file; all payload and prior receipts are covered.
    rows = {p.relative_to(destination).as_posix(): {'sha256':c.file_sha(p), 'bytes':p.stat().st_size}
            for p in list_files(destination) if p != destination / 'provenance/checksums.json'}
    write_json(destination / 'provenance/checksums.json', rows)
    return rows


def verify_archive(destination):
    destination = Path(destination)
    expected = c.read(destination / 'provenance/checksums.json')
    actual = {p.relative_to(destination).as_posix() for p in list_files(destination)
              if p != destination / 'provenance/checksums.json'}
    if set(expected) != actual:
        raise ValueError('Archive file inventory mismatch')
    for relative, row in expected.items():
        p = destination / relative
        if not p.resolve().is_relative_to(destination.resolve()):
            raise ValueError('Unsafe archived path')
        if c.file_sha(p) != row['sha256'] or p.stat().st_size != row['bytes']:
            raise ValueError(f'Archive integrity mismatch: {relative}')
    return {'files':len(expected), 'bytes':sum(r['bytes'] for r in expected.values()), 'verified':True}


def publish(staged, destination, *, desktop_instructions=None):
    staged, destination = Path(staged), Path(destination)
    verify_archive(staged)
    result = copy_checked([(p, p.relative_to(staged).as_posix()) for p in list_files(staged)], destination)
    checked = verify_archive(destination)
    if desktop_instructions is not None:
        target = Path(desktop_instructions)
        source = destination / '后续操作指令.md'
        shutil.copyfile(source, target)
        if c.file_sha(source) != c.file_sha(target):
            raise ValueError('Desktop instruction verification failed')
    return {**checked, 'copied_files':len(result), 'destination':str(destination)}
