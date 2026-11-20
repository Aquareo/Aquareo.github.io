#!/usr/bin/env python3
"""Install Cloudflare's public Web Analytics beacon on the personal homepages.

Read the dashboard's public JS snippet from stdin. No account API key is needed.
"""
import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path


class BeaconParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tokens = []
        self.module = False

    def handle_starttag(self, tag, attributes):
        attributes = dict(attributes)
        if tag != 'script' or attributes.get('src') != 'https://static.cloudflareinsights.com/beacon.min.js':
            return
        config = json.loads(attributes.get('data-cf-beacon', '{}'))
        token = config.get('token', '')
        if not isinstance(token, str) or not re.fullmatch(r'[0-9a-fA-F]{32}', token):
            raise ValueError('Expected the public 32-character Web Analytics site token.')
        self.tokens.append(token)
        self.module = attributes.get('type') == 'module'


def main():
    parser = BeaconParser()
    parser.feed(sys.stdin.read())
    if len(parser.tokens) != 1:
        raise ValueError('Paste exactly one Cloudflare Web Analytics JS snippet, not an API key.')
    config = json.dumps({'token': parser.tokens[0]}, separators=(',', ':'))
    loading = 'type="module"' if parser.module else 'defer'
    snippet = ('  <!-- Cloudflare Web Analytics -->\n'
               f'  <script {loading} src="https://static.cloudflareinsights.com/beacon.min.js" '
               f"data-cf-beacon='{config}'></script>\n"
               '  <!-- End Cloudflare Web Analytics -->\n')
    dist = Path(__file__).resolve().parents[1] / 'dist'
    changes = []
    for name in ('index.html', 'en/index.html', 'zh/index.html'):
        path = dist / name
        text = path.read_text()
        text = re.sub(r'  <!-- Cloudflare Web Analytics -->\n.*?  <!-- End Cloudflare Web Analytics -->\n', '', text, flags=re.S)
        if 'cloudflareinsights.com/beacon' in text:
            raise ValueError(f'Unmanaged analytics script in {name}; inspect before replacing.')
        if text.count('</body>') != 1:
            raise ValueError(f'Expected one closing body tag in {name}.')
        changes.append((path, text.replace('</body>', snippet + '</body>')))
    for path, text in changes:
        path.write_text(text)
        print(f'Configured {path.relative_to(dist)}')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, json.JSONDecodeError) as error:
        sys.exit(str(error))
