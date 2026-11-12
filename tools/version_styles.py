#!/usr/bin/env python3
"""Emit a content-versioned stylesheet and update both static language pages."""
from hashlib import sha256
from pathlib import Path
import re

def main():
    dist=Path(__file__).resolve().parents[1]/'dist'
    content=(dist/'styles.css').read_bytes()
    name='styles.'+sha256(content).hexdigest()[:12]+'.css'
    (dist/name).write_bytes(content)
    for relative in ('index.html','en/index.html','zh/index.html'):
        page=dist/relative
        text=page.read_text()
        updated,count=re.subn(r'(?P<prefix>(?:\.\./)?)styles(?:\.[0-9a-f]{12})?\.css(?:\?[^\"]*)?',lambda m:m.group('prefix')+name,text)
        if count!=1:
            raise ValueError(f'Expected one stylesheet link in {relative}; got {count}')
        page.write_text(updated)
    print(name)

if __name__=='__main__':
    main()
