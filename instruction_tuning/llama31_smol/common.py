"""Round 2 protocol: native Llama Base EOS and one shared train/eval serializer."""
import hashlib
import json
import os
from pathlib import Path
import re
import unicodedata
ROOT=Path(__file__).resolve().parent
ROUND=ROOT
REPO=ROOT.parents[1]
if not os.environ.get('UNILORA_STORAGE'):
    raise RuntimeError('Set UNILORA_STORAGE to a group-storage directory; source env.sh first.')
STORAGE=Path(os.environ['UNILORA_STORAGE']).expanduser().resolve()
CONFIG=json.loads((ROOT/'protocol.json').read_text())
MODEL=STORAGE/'models/llama31_base'
PREPARED=STORAGE/'datasets/i2_smolmagpie'

def sha256(path):
    d=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): d.update(chunk)
    return d.hexdigest()

def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');os.replace(tmp,path)

def normalize(text):
    return ' '.join(re.findall(r'\w+',unicodedata.normalize('NFKC',text).casefold()))

def group_key(messages):
    return hashlib.sha256(normalize(next((m['content'] for m in messages if m['role']=='user'),'')).encode()).hexdigest()

def validate_messages(messages):
    if not isinstance(messages,list) or not messages:return False
    if any(m.get('role') not in ('system','user','assistant') or not isinstance(m.get('content'),str) or not m['content'].strip() for m in messages):return False
    roles=[m['role'] for m in messages]
    if roles[0]=='system':roles=roles[1:]
    return bool(roles) and len(roles)%2==0 and all(r==('user' if i%2==0 else 'assistant') for i,r in enumerate(roles))

def header(role,first=False):
    return ('' if first else '\n')+'### '+role.capitalize()+':\n'

def serialize(messages,tokenizer,add_generation_prompt=False):
    """Tokenize header/body separately, identically for training and inference."""
    ids=[tokenizer.bos_token_id];labels=[-100]
    for i,m in enumerate(messages):
        h=tokenizer.encode(header(m['role'],i==0),add_special_tokens=False)
        body=tokenizer.encode(m['content'],add_special_tokens=False)
        ids+=h;labels += [-100]*len(h)
        ids+=body;labels += body if m['role']=='assistant' else [-100]*len(body)
        if m['role']=='assistant':ids.append(tokenizer.eos_token_id);labels.append(tokenizer.eos_token_id)
    if add_generation_prompt:
        assert messages and messages[-1]['role']=='user'
        h=tokenizer.encode(header('assistant'),add_special_tokens=False)
        ids+=h;labels += [-100]*len(h)
    return ids,labels

def encode_chat(messages,tokenizer,max_length=2048,include_audit=False):
    if not validate_messages(messages):return None
    if any(t in m['content'] for m in messages for t in tokenizer.all_special_tokens):return None
    ids,labels=serialize(messages,tokenizer)
    if len(ids)>=max_length:return None
    count=sum(x!=-100 for x in labels[1:])
    if not count:return None
    assert labels[-1]==tokenizer.eos_token_id
    assert labels.count(tokenizer.eos_token_id)==sum(m['role']=='assistant' for m in messages)
    row={'input_ids':ids,'labels':labels,'length':len(ids),'supervised_tokens':count,'truncated':False,'partial_answer':False}
    if include_audit:row.update(rendered_text=tokenizer.decode(ids),supervised_text=tokenizer.decode([x for x in labels if x!=-100]))
    return row

def load_tokenizer():
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True)
    assert tok.eos_token_id==128001 and tok.bos_token_id==128000
    tok.pad_token=tok.eos_token
    return tok

def collate(rows,pad_id,device):
    import torch
    n=max(len(r['input_ids']) for r in rows)
    ids=torch.full((len(rows),n),pad_id,dtype=torch.long,device=device)
    labels=torch.full_like(ids,-100);mask=torch.zeros_like(ids)
    for i,r in enumerate(rows):
        length=len(r['input_ids']);ids[i,:length]=torch.tensor(r['input_ids'],device=device)
        labels[i,:length]=torch.tensor(r['labels'],device=device);mask[i,:length]=1
    return {'input_ids':ids,'attention_mask':mask,'labels':labels}
