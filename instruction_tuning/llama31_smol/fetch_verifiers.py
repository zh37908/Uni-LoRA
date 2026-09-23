"""Vendor only the pinned official evaluation files needed by I1."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import urllib.request

root = Path(os.environ['UNILORA_STORAGE'])
(root/'manifests').mkdir(parents=True,exist_ok=True)
external = Path(os.environ['UNILORA_STORAGE']) / 'external'
external.mkdir(exist_ok=True)
google_rev = '08a8d6736475776f42ffac23b2c13111a28e5795'
ifbench_rev = '1c40f0c10d9b5c5c2f10a175a28007ebb64f7f4d'
op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
def get(url):
    with op.open(url, timeout=30) as response:
        return response.read()
target = external / 'google_ifeval' / 'instruction_following_eval'
target.mkdir(parents=True, exist_ok=True)
items = json.loads(get('https://api.github.com/repos/google-research/google-research/contents/'
                      f'instruction_following_eval?ref={google_rev}'))
def download(item):
    if item['type'] != 'file': return None
    data = get(item['download_url'])
    (target / item['name']).write_bytes(data)
    return {'name': item['name'], 'sha256': hashlib.sha256(data).hexdigest()}
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    files = [x for x in executor.map(download, items) if x]
checkout = external / 'IFBench'
if not checkout.exists():
    subprocess.run(['git', 'clone', '--quiet', 'https://github.com/allenai/IFBench.git', str(checkout)], check=True)
subprocess.run(['git', '-C', str(checkout), 'checkout', '--quiet', ifbench_rev], check=True)
manifest = {'ifeval': {'repository': 'google-research/google-research', 'commit': google_rev,
                       'files': files, 'path': str(target.parent)},
            'ifbench': {'repository': 'allenai/IFBench', 'commit': ifbench_rev, 'path': str(checkout)}}
(root / 'manifests/verifier_sources.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(manifest, indent=2))
