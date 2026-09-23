"""Fetch pinned public artifacts into explicit group storage; never Git."""
import json,subprocess,sys,urllib.request
from pathlib import Path
from huggingface_hub import snapshot_download
from common import CONFIG,ROOT,STORAGE,sha256

def main():
    for name,kind,repo,revision,patterns,parent in [
        ('llama31_base','model',CONFIG['model_id'],CONFIG['model_revision'],['*.json','*.safetensors','tokenizer.model','LICENSE','USE_POLICY.md'],'models'),
        ('smoltalk','dataset',CONFIG['dataset_id'],CONFIG['dataset_revision'],['data/all/*.parquet'],'datasets'),
    ]:
        path=Path(snapshot_download(repo,repo_type=kind,revision=revision,allow_patterns=patterns,ignore_patterns=['original/*'],max_workers=4))
        assert path.resolve().is_relative_to(STORAGE.resolve())
        link=STORAGE/parent/name;link.parent.mkdir(parents=True,exist_ok=True)
        if not link.exists():link.symlink_to(path,target_is_directory=True)
        assert link.resolve()==path.resolve()
    subprocess.run([sys.executable,str(ROOT/'fetch_verifiers.py')],check=True)
    expected=json.loads((ROOT/'expected_data.json').read_text())['benchmark_hashes']
    urls={
        'ifeval':'https://raw.githubusercontent.com/google-research/google-research/08a8d6736475776f42ffac23b2c13111a28e5795/instruction_following_eval/data/input_data.jsonl',
        'ifbench':'https://raw.githubusercontent.com/allenai/IFBench/1c40f0c10d9b5c5c2f10a175a28007ebb64f7f4d/ifbench/data/IFBench_test.jsonl',
    }
    for name,url in urls.items():
        f=STORAGE/'datasets/benchmarks'/f'{name}.jsonl';f.parent.mkdir(parents=True,exist_ok=True)
        if not f.exists():
            with urllib.request.urlopen(url,timeout=60) as response:f.write_bytes(response.read())
        assert sha256(f)==expected[f.name],f'Unexpected benchmark bytes: {f}'
    import nltk
    for resource in ['punkt','punkt_tab']:assert nltk.download(resource,quiet=True)

if __name__=='__main__':main()
