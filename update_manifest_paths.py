#!/usr/bin/env python3
"""
Update manifest file paths for a new location.
Replaces the old path prefix with a new one in all manifests.
"""

import json
import argparse
from pathlib import Path

def update_manifest_paths(manifest_path: str, old_prefix: str, new_prefix: str, backup: bool = True):
    """Update audio file paths in a manifest file."""
    manifest_path = Path(manifest_path)

    # Create backup
    if backup:
        backup_path = manifest_path.with_suffix('.jsonl.bak')
        print(f"Creating backup: {backup_path}")
        import shutil
        shutil.copy2(manifest_path, backup_path)

    # Read all lines
    with open(manifest_path, 'r') as f:
        lines = f.readlines()

    # Update paths
    updated_lines = []
    updated_count = 0

    for line in lines:
        data = json.loads(line.strip())
        old_path = data['audio_filepath']

        if old_prefix in old_path:
            new_path = old_path.replace(old_prefix, new_prefix)
            data['audio_filepath'] = new_path
            updated_count += 1

        updated_lines.append(json.dumps(data))

    # Write updated manifest
    with open(manifest_path, 'w') as f:
        for line in updated_lines:
            f.write(line + '\n')

    print(f"✓ Updated {updated_count}/{len(lines)} paths in {manifest_path}")
    return updated_count

def main():
    parser = argparse.ArgumentParser(
        description='Update manifest paths for a new location',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Update all manifests in output/manifests/
  python update_manifest_paths.py \\
    --old-prefix /Users/yaoyu/projects/speech_finetune_data_generation \\
    --new-prefix /home/user/finetuning \\
    --manifest-dir output/manifests

  # Update single manifest
  python update_manifest_paths.py \\
    --old-prefix /old/path \\
    --new-prefix /new/path \\
    --manifest output/manifests/train_manifest.jsonl
        '''
    )

    parser.add_argument('--old-prefix', required=True,
                        help='Old path prefix to replace')
    parser.add_argument('--new-prefix', required=True,
                        help='New path prefix')
    parser.add_argument('--manifest',
                        help='Single manifest file to update')
    parser.add_argument('--manifest-dir',
                        help='Directory containing manifests (updates train, val, master)')
    parser.add_argument('--no-backup', action='store_true',
                        help='Skip creating backup files')

    args = parser.parse_args()

    if not args.manifest and not args.manifest_dir:
        parser.error('Must specify either --manifest or --manifest-dir')

    total_updated = 0

    if args.manifest:
        # Update single manifest
        total_updated += update_manifest_paths(
            args.manifest,
            args.old_prefix,
            args.new_prefix,
            backup=not args.no_backup
        )

    if args.manifest_dir:
        # Update all manifests in directory
        manifest_dir = Path(args.manifest_dir)
        for manifest_name in ['train_manifest.jsonl', 'val_manifest.jsonl', 'master_manifest.jsonl']:
            manifest_path = manifest_dir / manifest_name
            if manifest_path.exists():
                total_updated += update_manifest_paths(
                    str(manifest_path),
                    args.old_prefix,
                    args.new_prefix,
                    backup=not args.no_backup
                )
            else:
                print(f"⚠ Skipping {manifest_path} (not found)")

    print(f"\n✓ Total paths updated: {total_updated}")

if __name__ == '__main__':
    main()
