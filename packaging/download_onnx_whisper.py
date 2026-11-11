"""Fetch the fixed upstream Whisper ONNX files using saved mirror/proxy settings."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from utau_forge.resource_download import fetch, load_preferences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--proxy', help='HTTP/HTTPS proxy, for example http://127.0.0.1:7890')
    parser.add_argument('--mirror', help='Hugging Face mirror base URL')
    args = parser.parse_args()
    preferences = load_preferences()
    if args.proxy:
        preferences['proxy'] = args.proxy
    if args.mirror:
        preferences.update(source='mirror', hf_mirror=args.mirror)
    records = json.loads((ROOT/'packaging/whisper_onnx_sources.json').read_text(encoding='utf-8'))
    progress = [0]
    def emit(**event):
        current = event.get('current',0)
        if current-progress[0] >= 16*1024**2 or event.get('stage') == '校验下载':
            progress[0] = current
            print(event,flush=True)
    for item in records['files']:
        print('Preparing',item['filename'],flush=True)
        progress[0] = 0
        fetch({'cache':item['filename'], 'url':item['url'], 'size':item['size'], 'sha256':item['sha256'],
               'range_size':8*1024**2},args.output.resolve(),preferences,None,emit)
    print(args.output.resolve(),flush=True)


if __name__ == '__main__':
    main()
