"""Fair GLUE protocol around a frozen copy of the existing training driver.
Train-only 90/10 split selects hyperparameters and checkpoints. Original dev
is evaluated once only in final phase. No shared training source is modified.
"""
from pathlib import Path
import argparse, hashlib, importlib.util, json, os, sys, time
P=Path(__file__).resolve().parent;R=P.parents[2]
sys.path.insert(0,str(R/'NLU/peft/src'))
ap=argparse.ArgumentParser();ap.add_argument('--spec',required=True);a=ap.parse_args()
spec=json.loads(Path(a.spec).read_text());out=P/'results'/spec['id'];out.mkdir(parents=True,exist_ok=True)
if (out/'result.json').exists():print('Already complete',spec['id']);sys.exit(0)
if spec['phase']=='final' and not spec.get('smoke'):
    locked=json.loads((P/'locked_selection.json').read_text())
    assert locked[spec['task']+'/'+spec['method']]['config_id']==spec['config_id']

os.environ['TOKENIZERS_PARALLELISM']='false'
import numpy as np,torch
from datasets import DatasetDict
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
loader=importlib.util.spec_from_file_location('revision_glue',P/'snapshots/glue_driver.py');m=importlib.util.module_from_spec(loader);loader.loader.exec_module(m)
load_data=m.load_dataset;original_eval=m.evaluate_glue_model
heldout={};record={};started=time.time()
def revised_data(name,task,*args,**kwargs):
    ds=load_data(name,task,*args,**kwargs)
    # Keep official dev out of every tuning/early-stopping decision.
    ds['train']=ds['train'].add_column('revision_row_id',list(range(len(ds['train']))))
    split=ds['train'].train_test_split(test_size=.1,seed=20260913,stratify_by_column='label')
    ids={k:split[k]['revision_row_id'] for k in ['train','test']}
    assert not(set(ids['train'])&set(ids['test']))
    record['split']={k:dict(n=len(v),sha256=hashlib.sha256(np.asarray(v,dtype='<i8').tobytes()).hexdigest()) for k,v in ids.items()}
    heldout['raw']=split['test'].remove_columns('revision_row_id') if spec.get('smoke') else ds['validation']
    return DatasetDict(train=split['train'].remove_columns('revision_row_id'),validation=split['test'].remove_columns('revision_row_id'))
m.load_dataset=revised_data
# Identical checkpoint eligibility across methods: after the shared support
# warmup/scoring budget, preventing restoration of a partly initialized mask.
original_adam=m.AdamW
class CountedAdamW(original_adam):
    def step(self,*args,**kwargs):
        result=super().step(*args,**kwargs)
        record['optimizer_steps']=record.get('optimizer_steps',0)+1
        return result
m.AdamW=CountedAdamW

def evaluated(model,eval_loader,task,metric_name,device,return_logits=False):
    if return_logits:
        # First evaluate the restored (or explicitly last) checkpoint on tuning
        # data, then on official dev only for locked final specifications.
        sel=original_eval(model,eval_loader,task,metric_name,device,return_logits=True)
        record['selection_loss']=sel[0];record['selection_metrics']=sel[1];record['selection_score']=sel[2]
        record['trainable_total']=sum(p.numel() for p in model.parameters() if p.requires_grad)
        record['trainable_adapter']=sum(p.numel() for n,p in model.named_parameters() if p.requires_grad and 'classifier' not in n)
        record['effective_adapter_coordinates']=record['trainable_adapter']
        if spec['method']=='prolosa':
            backend=m.get_unilora_rosa_backend(model,variant)
            stats=backend.get_sparse_structure_stats()
            dense_sparse=sum(p.numel() for n,p in model.named_parameters() if p.requires_grad and 'unilora_rosa_sparse_theta_D' in n)
            record['effective_adapter_coordinates']-=dense_sparse-int(stats['selected_positions'])
            record['sparse_stats']=stats
        expected=23040 if spec['method'] in ['unilora','prolosa'] else 1769472
        assert record['effective_adapter_coordinates']==expected,(spec['method'],record)
        record['heldout_is_smoke_surrogate']=bool(spec.get('smoke'))
        if spec['phase']=='final':
            tokenizer=AutoTokenizer.from_pretrained('roberta-large',use_fast=True)
            k1,k2=m.TASK_TO_KEYS[task]
            def tok(batch):
                return tokenizer(batch[k1],None if k2 is None else batch[k2],truncation=True,padding='max_length',max_length=m.MAX_LENGTH['roberta-large'])
            raw=heldout['raw'];ds=raw.map(tok,batched=True,remove_columns=[c for c in raw.column_names if c!='label']).rename_column('label','labels')
            dl=DataLoader(ds,batch_size=32,shuffle=False,collate_fn=lambda xs:tokenizer.pad(xs,return_tensors='pt'),num_workers=0)
            val=original_eval(model,dl,task,metric_name,device,return_logits=True)
            record['heldout_loss']=val[0];record['heldout_metrics']=val[1];record['heldout_score']=val[2]
            np.savez_compressed(out/'heldout_predictions.npz',logits=val[3].cpu().numpy(),labels=np.asarray(raw['label']))
            return val
        return sel
    result=original_eval(model,eval_loader,task,metric_name,device,return_logits=return_logits)
    if record.get('optimizer_steps',0)<(4 if spec.get('smoke') else 256):
        return result[0],result[1],-1e18
    return result
m.evaluate_glue_model=evaluated
if spec['stopping']=='last':m.restore_trainable_state=lambda model,state:None
method=spec['method'];rank=2 if method=='rosa' else 4
variant={'lora':'lora','unilora':'unilora','prolosa':'unilora_rosa_snip','rosa':'lora_rosa'}[method]
argv=['glue_driver.py','--variant',variant,'--model_name','roberta-large','--task',spec['task'],
      '--rank',str(rank),'--lora_alpha',str(rank),'--seed',str(spec['seed']),'--proj_seed','13',
      '--num_epochs','40','--batch_size','32','--head_lr',str(spec['head_lr']),
      '--lora_lr',str(spec['adapter_lr']),'--theta_d_lr',str(spec['adapter_lr']),
      '--unilora_dropout',str(spec['dropout']),'--weight_decay',str(spec['weight_decay']),
      '--rosa_sparse_lr',str(spec['adapter_lr']*.2),'--rosa_grad_acc_mode','mean_squared',
      '--rosa_warmup_steps','128','--rosa_mask_steps','128','--out_dir',str(out),'--save_eval_logits']
if method=='unilora':argv+=['--theta_d_length','23040']
if method=='prolosa':
    # ceil(D*density) is used by this implementation. A midpoint avoids a
    # float-induced off-by-one while giving exactly K=720 actual mask entries.
    argv+=['--theta_d_length','22320','--rosa_density',str((720-.5)/1769472)]
if method=='rosa':argv+=['--rosa_sparse_budget','884736']
if spec.get('smoke'):
    argv+=['--max_steps','12','--eval_every_steps','6','--rosa_warmup_steps','2','--rosa_mask_steps','2']
sys.argv=argv
m.main()
assert 'selection_score' in record
assert ('heldout_score' in record)==(spec['phase']=='final')
record.update(spec=spec,elapsed_seconds=time.time()-started,gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
              peak_gpu_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
              driver_sha256=hashlib.sha256((P/'snapshots/glue_driver.py').read_bytes()).hexdigest())
(out/'result.json').write_text(json.dumps(record,indent=2)+'\n');print('COMPLETE',spec['id'],record['elapsed_seconds'],flush=True)
