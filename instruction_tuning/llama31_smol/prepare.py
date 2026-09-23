"""Select complete short smol-magpie conversations; never truncate answers."""
import collections,hashlib,json,sys,time
from datasets import Dataset
from common import *
from data_utils import PromptFilter, rows

def main():
    if (PREPARED/'manifest.json').exists():raise RuntimeError('Frozen dataset already exists')
    PREPARED.mkdir(parents=True,exist_ok=True)
    tok=load_tokenizer()
    files=[STORAGE/'datasets/benchmarks'/f'{b}.jsonl' for b in ['ifeval','ifbench']]
    checker=PromptFilter([json.loads(l)['prompt'] for p in files for l in p.read_text().splitlines()])
    selected={};excluded={};selected_keys=set();audit=[]
    for source_split in ['train','test']:
        candidates=[];seen=set();stats=collections.Counter()
        for i,row in rows(source_split):
            if row['source']!=CONFIG['source']:continue
            stats['source_rows']+=1
            messages=row['messages']
            if not validate_messages(messages):stats['bad_format']+=1;continue
            key=group_key(messages)
            if key in seen or key in selected_keys:stats['duplicate']+=1;continue
            seen.add(key)
            if any(checker.overlaps(m['content']) for m in messages if m['role']=='user'):stats['benchmark_overlap']+=1;continue
            rank=hashlib.sha256(f"{source_split}:{CONFIG['split_seed']}:{key}".encode()).digest()
            candidates.append((rank,{'row_id':f'{source_split}:{i}','group':key,'source':row['source'],'messages':messages}))
            if stats['source_rows']%20000==0:print('SCAN',source_split,i,dict(stats),flush=True)
        candidates.sort(key=lambda pair:pair[0])
        need=CONFIG['train_size']+CONFIG['val_size'] if source_split=='train' else CONFIG['test_size']
        accepted=[]
        for _,row in candidates:
            enc=encode_chat(row['messages'],tok,CONFIG['max_length'])
            if enc is None:stats['length_or_reserved_or_empty']+=1;continue
            accepted.append((row,enc))
            if len(accepted)%5000==0:print('ACCEPTED',source_split,len(accepted),flush=True)
            if len(accepted)==need:break
        assert len(accepted)==need,(source_split,len(accepted),need)
        selected_keys.update(r['group'] for r,e in accepted)
        if source_split=='train':
            selected['train']=accepted[:CONFIG['train_size']];selected['val']=accepted[CONFIG['train_size']:]
        else:selected['test']=accepted
        excluded[source_split]=dict(stats)
        del candidates
    manifest={'config':CONFIG,'config_sha256':sha256(ROUND/'protocol.json'),'prepare_sha256':sha256(__file__),
      'serializer_sha256':sha256(ROUND/'common.py'),'benchmark_hashes':{p.name:sha256(p) for p in files},
      'selection':'source-filter; first-user-dedup; benchmark prompt decontamination; SHA256 sorted candidates; reject whole length>=2048; first 50K train then 1K val; separate official test 1K',
      'excluded':excluded,'splits':{},'eos_token_id':tok.eos_token_id}
    for split,data in selected.items():
        path=PREPARED/f'{split}.jsonl'
        with path.open('w') as f:
            for row,e in data:f.write(json.dumps(row,ensure_ascii=False)+'\n')
        ds=Dataset.from_list([{**{k:v for k,v in row.items() if k!='messages'},**e} for row,e in data])
        ds.save_to_disk(str(PREPARED/split))
        manifest['splits'][split]={'examples':len(ds),'raw_sha256':sha256(path),'fingerprint':ds._fingerprint,'tokens':sum(e['length'] for r,e in data),'supervised_tokens':sum(e['supervised_tokens'] for r,e in data),'eos_labels':sum(e['labels'].count(tok.eos_token_id) for r,e in data),'max_length':max(e['length'] for r,e in data)}
        audit.extend({'split':split,'row_id':r['row_id'],**encode_chat(r['messages'],tok,CONFIG['max_length'],True)} for r,e in data[:3])
    for a,b in [('train','val'),('train','test'),('val','test')]:
        assert not ({r['group'] for r,e in selected[a]} & {r['group'] for r,e in selected[b]})
    write_json(PREPARED/'token_mask_audit.json',audit);write_json(PREPARED/'manifest.json',manifest)
    write_json(STORAGE/'manifests/data.json',manifest)
    print('PREPARATION_COMPLETE',json.dumps(manifest['splits']),flush=True)
if __name__=='__main__':main()
