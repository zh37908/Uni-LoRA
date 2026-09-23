"""Synthetic instruction development diagnostics. See README for actual selection protocol."""
import json,re,sys,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
from common import STORAGE, PREPARED, serialize, write_json, sha256
DEV=STORAGE/'datasets/i3_development'
TOPICS=['keeping a tidy desk','planning a weekend walk','learning a new language','preparing a simple breakfast','caring for a houseplant','organizing a reading group','packing for a short trip','starting a daily journal','saving water at home','practicing a musical instrument','planning a community picnic','building a regular study habit']

def build_records():
 records=[]
 for i,topic in enumerate(TOPICS):
  prompts={
   'json':f'Give one practical tip about {topic}. Reply only with a JSON object containing exactly two keys: "topic" and "tip". Set "topic" to "{topic}" and "tip" to a nonempty string. Do not use a code block or any text outside the JSON object.',
   'words':f'Write exactly 40 whitespace-separated words of advice about {topic}. Put the entire response on one line.',
   'bullets':f'Give four suggestions about {topic}. Your entire response must have exactly four nonempty lines. Start every line with "- " followed by the suggestion. Include no title or other lines.',
   'numbered':f'Explain {topic} in exactly three steps. Your entire response must have three nonempty lines, starting with "1. ", "2. ", and "3. " respectively. Put one step on each line and no other text.',
   'upper':f'Give advice about {topic} in at least 20 words. Write every alphabetic character in UPPERCASE.',
   'lower':f'Give advice about {topic} in at least 20 words. Write every alphabetic character in lowercase. Include the standalone word "plan" exactly once.',
   'bounds':f'Give advice about {topic}. Start your response with a line containing only START and end it with a line containing only END. Between those lines, write at least 20 words of advice.',
   'keywords':f'Write at least 30 words of advice about {topic}. Include each of the standalone words "plan", "today", and "review". Do not include the word "perfect" in any capitalization.'}
  for rule,prompt in prompts.items():records.append({'id':f'{i:02d}_{rule}','topic':topic,'rule':rule,'prompt':prompt})
 return records

def passes(row,text):
 text=text.strip();rule=row['rule'];words=re.findall(r'\b[a-z]+\b',text.lower());lines=text.splitlines()
 if rule=='json':
  try:d=json.loads(text)
  except (ValueError,TypeError):return False
  return isinstance(d,dict) and set(d)=={'topic','tip'} and d['topic']==row['topic'] and isinstance(d['tip'],str) and bool(d['tip'].strip())
 if rule=='words':return len(text.split())==40 and len(lines)==1
 if rule=='bullets':return len(lines)==4 and all(re.fullmatch(r'- \S.*',s) for s in lines)
 if rule=='numbered':return len(lines)==3 and all(s.startswith(f'{i+1}. ') and s[3:].strip() for i,s in enumerate(lines))
 if rule=='upper':return len(text.split())>=20 and text==text.upper() and bool(words)
 if rule=='lower':return len(text.split())>=20 and text==text.lower() and words.count('plan')==1
 if rule=='bounds':return len(lines)>=3 and lines[0]=='START' and lines[-1]=='END' and len('\n'.join(lines[1:-1]).split())>=20
 if rule=='keywords':return len(text.split())>=30 and all(w in words for w in ['plan','today','review']) and 'perfect' not in words
 raise ValueError(rule)

def prepare():
 from data_utils import PromptFilter
 records=build_records();assert len(records)==96
 checker=PromptFilter([r['prompt'] for r in records]);overlaps=[]
 with (PREPARED/'train.jsonl').open() as f:
  for line in f:
   row=json.loads(line)
   if any(checker.overlaps(m['content']) for m in row['messages'] if m['role']=='user'):overlaps.append(row['row_id'])
 assert not overlaps,f'Development/train overlap: {overlaps[:10]}'
 DEV.mkdir(parents=True,exist_ok=True)
 target=DEV/'prompts.jsonl'
 text=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records)
 if target.exists():assert target.read_text()==text
 else:target.write_text(text)
 write_json(DEV/'manifest.json',{'n':96,'source':'predeclared local synthetic prompts, not benchmark examples','max_new_tokens':256,'train_overlap_rows':overlaps,'train_raw_sha256':sha256(PREPARED/'train.jsonl'),'prompts_sha256':sha256(target),'scorer_sha256':sha256(__file__)})
 print('DEVELOPMENT_SET_READY',flush=True)

def evaluate(model,tok,out):
 import torch,time
 manifest=json.loads((DEV/'manifest.json').read_text())
 assert manifest['scorer_sha256']==sha256(__file__)
 assert manifest['prompts_sha256']==sha256(DEV/'prompts.jsonl')
 records=[json.loads(l) for l in (DEV/'prompts.jsonl').read_text().splitlines()]
 out=Path(out);out.mkdir(parents=True,exist_ok=True)
 f=out/'dev_predictions.jsonl';done=[json.loads(l) for l in f.read_text().splitlines()] if f.exists() else []
 assert [r['id'] for r in done]==[r['id'] for r in records[:len(done)]]
 was_training=model.training;cache=model.config.use_cache;model.eval();model.config.use_cache=True
 with f.open('a') as handle:
  for row in records[len(done):]:
   ids,_=serialize([{'role':'user','content':row['prompt']}],tok,True)
   ids=torch.tensor([ids],device=next(model.parameters()).device);tick=time.monotonic()
   with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
    output=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),max_new_tokens=256,do_sample=False,temperature=None,top_p=None,top_k=None,eos_token_id=tok.eos_token_id,pad_token_id=tok.pad_token_id)
   new=output[0,ids.shape[1]:].tolist();eos=bool(new and new[-1]==tok.eos_token_id)
   answer=tok.decode(new[:-1] if eos else new,skip_special_tokens=False)
   rec={**row,'response':answer,'pass':bool(passes(row,answer)),'eos':eos,'generated_tokens':len(new),'seconds':time.monotonic()-tick}
   handle.write(json.dumps(rec,ensure_ascii=False)+'\n');handle.flush();done.append(rec)
   if len(done)%16==0:print('DEV_GENERATED',len(done),flush=True)
 report={'n':len(done),'passed':sum(r['pass'] for r in done),'accuracy':sum(r['pass'] for r in done)/len(done),'eos_count':sum(r['eos'] for r in done),'empty':sum(not r['response'].strip() for r in done),'manifest':manifest,'predictions_sha256':sha256(f)}
 write_json(out/'dev_metrics.json',report);model.config.use_cache=cache;model.train(was_training)
 return report
if __name__=='__main__':prepare()
