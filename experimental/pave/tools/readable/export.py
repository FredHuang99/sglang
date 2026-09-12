"""Rebuild the completed paper's readable package from a lossless archive.

This exporter deliberately validates the final 342-run paper layout. It is not
an automatic selector or an exporter for an arbitrary new experiment matrix.
"""
import argparse
import csv
import json
import runpy
import tempfile
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    archive = args.archive_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    code = Path(__file__).resolve().parent
    if not (archive / 'provenance/checksums.json').is_file():
        parser.error('Expected a complete PAVE results archive with checksums')
    if output.exists():
        parser.error('Output must be a new directory; existing data are never overwritten')
    if output.is_relative_to(archive) or output.is_relative_to(code.parents[1]):
        parser.error('Choose an output outside both the archive and the PAVE source tree')
    if __debug__ is False:
        parser.error('Do not use python -O: independent data assertions are required')
    with tempfile.TemporaryDirectory(prefix='pave-readable-') as work:
        context = dict(archive=str(archive), output=str(output), work=work)
        try:
            runpy.run_path(str(code / '_build.py'), init_globals={'REBUILD_CONTEXT': context})
            specs = json.loads((Path(work) / 'csv_specs.json').read_text('utf-8'))
            for spec in specs:
                path = output / spec['path']
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('w', encoding='utf-8-sig', newline='') as stream:
                    writer = csv.writer(stream)
                    for row in spec['matrix']:
                        writer.writerow([str(v).lower() if isinstance(v, bool) else v for v in row])
            runpy.run_path(str(code / '_validate.py'), init_globals={'REBUILD_CONTEXT': context})
        except BaseException:
            if output.exists():
                (output / 'INCOMPLETE.txt').write_text(
                    'Export did not finish verification. Do not use as a validated publication.\n',
                    encoding='utf-8')
            raise
    print(json.dumps({'output_dir': str(output), 'verified': True, 'simulation_runs_added': 0}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
