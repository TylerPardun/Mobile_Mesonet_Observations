from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parent
manifest=json.loads((root/'MANIFEST.json').read_text())
for name, expected in manifest['sha256'].items():
    actual=hashlib.sha256((root/name).read_bytes()).hexdigest()
    if actual != expected: raise SystemExit(f'Hash mismatch: {name}')
print(f"Verified {len(manifest['sha256'])} release files.")
