import argparse
import json
from pathlib import Path


def main(argv=None):
    p = argparse.ArgumentParser(prog='python -m evaluation_tools')
    sub = p.add_subparsers(dest='command', required=True)
    revision = sub.add_parser('revise-cluster')
    revision.add_argument('--campaign', type=Path, required=True)
    revision.add_argument('--evidence', type=Path, required=True)
    run = sub.add_parser('run')
    run.add_argument('--campaign', type=Path, required=True)
    run.add_argument('--plan', type=Path, required=True)
    run.add_argument('--batch', choices=['f-low','f-rest','d','e'], required=True)
    run.add_argument('--retry-failed', action='store_true')
    paper = sub.add_parser('paper')
    paper.add_argument('--campaign', type=Path, required=True)
    paper.add_argument('--output-dir', type=Path, required=True)
    paper.add_argument('--experiments', nargs='+', choices=list('abc'), default=['a','b','c'])
    archive = sub.add_parser('archive')
    archive.add_argument('--campaign', type=Path, required=True)
    archive.add_argument('--output-dir', type=Path, required=True)
    archive.add_argument('--evidence', type=Path)
    archive.add_argument('--source-dir', type=Path)
    archive.add_argument('--original-trace', type=Path)
    verify = sub.add_parser('verify-archive')
    verify.add_argument('--directory', type=Path, required=True)
    pub = sub.add_parser('publish')
    pub.add_argument('--staged', type=Path, required=True)
    pub.add_argument('--destination', type=Path, required=True)
    pub.add_argument('--desktop-instructions', type=Path)
    args = p.parse_args(argv)
    if args.command == 'revise-cluster':
        from .selection import revise_cluster
        result = revise_cluster(args.campaign, args.evidence)
    elif args.command == 'run':
        from .runner import run_batch
        result = run_batch(args.campaign, args.batch, args.plan, retry_failed=args.retry_failed)
    elif args.command == 'paper':
        from .publication import export_paper
        result = export_paper(args.campaign, args.output_dir, args.experiments)
    elif args.command == 'archive':
        from .archive import assemble
        result = assemble(args.campaign, args.output_dir, args.evidence,
                          source_dir=args.source_dir, original_trace=args.original_trace)
    elif args.command == 'publish':
        from .archive import publish
        result = publish(args.staged, args.destination, desktop_instructions=args.desktop_instructions)
    else:
        from .archive import verify_archive
        result = verify_archive(args.directory)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
