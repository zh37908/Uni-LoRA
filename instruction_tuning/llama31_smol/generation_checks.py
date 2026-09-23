"""Small non-benchmark generation diagnostics, using the training serializer."""
import json,time
import torch
from common import *
@torch.no_grad()
def check_generation(model,tokenizer,out,stage,require_eos=False):
    was_training=model.training;old_cache=model.config.use_cache
    model.eval();model.config.use_cache=True
    prompts=[('simple:0',[{'role':'user','content':'Reply with only the word Hello.'}]),
             ('simple:1',[{'role':'user','content':'What is 2 + 2? Answer with only the number.'}])]
    with (PREPARED/'val.jsonl').open() as f:
        for line in f:
            row=json.loads(line)
            # Held-out multi-turn context, with a short final reference answer.
            if len(row['messages'][-1]['content'])<300:
                prompts.append((row['row_id'],row['messages'][:-1]))
                if len(prompts)==6:break
    results=[]
    for key,messages in prompts:
        ids,_=serialize(messages,tokenizer,True)
        ids=torch.tensor([ids],device=next(model.parameters()).device)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            generated=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),max_new_tokens=256,
                do_sample=False,temperature=None,top_k=None,top_p=None,eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.pad_token_id)
        new=generated[0,ids.shape[1]:].tolist()
        results.append({'key':key,'prompt':messages,'tokens':len(new),'eos':bool(new and new[-1]==tokenizer.eos_token_id),
                        'response':tokenizer.decode(new,skip_special_tokens=False)})
    report={'stage':stage,'count':len(results),'eos_count':sum(r['eos'] for r in results),'max_new_tokens':256,'results':results}
    write_json(out/f'generation_{stage}.json',report)
    print('GENERATION_CHECK',stage,report['eos_count'],'/',report['count'],flush=True)
    model.config.use_cache=old_cache;model.train(was_training)
    if require_eos and report['eos_count']==0:raise RuntimeError('All generation diagnostic samples hit length limit; checkpoint preserved, inspect before continuation')
    return report
