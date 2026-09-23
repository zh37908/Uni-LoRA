"""Single-GPU I1 training with token-weighted loss and complete adapter resume."""
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import resource
import shutil
import signal
import subprocess
import time

import numpy as np
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig, LlamaForCausalLM
from peft import LoraConfig, UniLoRAConfig, UniLoRARoSASnipConfig, get_peft_model
from common import REPO, CONFIG, ROOT, ROUND, STORAGE, MODEL, PREPARED, collate, sha256, write_json, load_tokenizer
from generation_checks import check_generation
assert Path(__import__('peft').__file__).resolve().is_relative_to((REPO/'math_instruction_tuning/peft/src').resolve()), 'Expected frozen original max PEFT'

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def build_model(method, cfg, tiny=False):
    seed_all(cfg['seed'])
    if tiny:
        base = LlamaForCausalLM(LlamaConfig(vocab_size=128, hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
            max_position_embeddings=256, attention_dropout=0.0))
    else:
        base = AutoModelForCausalLM.from_pretrained(str(MODEL), local_files_only=True,
            torch_dtype=torch.bfloat16, attn_implementation='sdpa')
    targets = cfg['targets']
    D = sum(cfg['rank'] * (m.in_features + m.out_features) for n, m in base.named_modules()
            if isinstance(m, torch.nn.Linear) and n.split('.')[-1] in targets)
    common = dict(r=cfg['rank'], target_modules=targets, task_type='CAUSAL_LM', bias='none')
    if method == 'lora':
        adapter = LoraConfig(**common, lora_alpha=cfg['lora_alpha'], lora_dropout=0)
    elif method == 'unilora':
        adapter = UniLoRAConfig(**common, vector_length=cfg['budget'], num_vectors=2048,
            unilora_dropout=0, init_vector_bank_bound=cfg['init_bound'], save_only_topk_weights=False)
    else:
        adapter = UniLoRARoSASnipConfig(**common, theta_d_length=cfg['d'], proj_seed=cfg['seed'],
            init_theta_d_bound=cfg['init_bound'], unilora_dropout=0,
            rosa_density=cfg['k']/D, rosa_warmup_steps=cfg['warmup_steps'], rosa_mask_steps=cfg['score_steps'])
    model = get_peft_model(base, adapter)
    # Explicitly standardize optimizer master parameter precision.
    for name, p in model.named_parameters():
        if 'lora_' in name: p.data = p.data.float()
    model.config.use_cache = False
    if not tiny:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.enable_input_require_grads()
    return model, D

def backend(model, method):
    return model.base_model if method == 'prolosa' else None

def adapter_tensors(model):
    return {n: t for n, t in list(model.named_parameters()) + list(model.named_buffers()) if 'lora_' in n}

def optimizer_for(model, cfg, method):
    main, sparse = [], []
    for name, p in model.named_parameters():
        if 'unilora_rosa_sparse_theta_D' in name: sparse.append(p)
        elif p.requires_grad: main.append(p)
    groups = [{'params':main, 'lr':cfg['lrs'][method], 'kind':'main'}]
    if sparse: groups.append({'params':sparse, 'lr':cfg['lrs'][method]*cfg['sparse_lr_mult'], 'kind':'sparse'})
    default_lr = cfg['prolosa_default_lr'] if method == 'prolosa' else cfg['lrs'][method]
    return torch.optim.AdamW(groups, lr=default_lr, betas=(.9,.999), eps=1e-8, weight_decay=0)

def set_lrs(optimizer, step, total_steps, cfg, method):
    warmup = math.ceil(total_steps * cfg['scheduler_warmup_ratio'])
    activation = cfg['warmup_steps'] + cfg['score_steps']
    for group in optimizer.param_groups:
        if group['kind'] == 'sparse':
            progress = max(0., (step-activation)/max(1,total_steps-activation))
            group['lr'] = cfg['lrs'][method]*cfg['sparse_lr_mult']*.5*(1+math.cos(math.pi*min(1.,progress)))
        else:
            factor = step/max(1,warmup) if step < warmup else .5*(1+math.cos(math.pi*(step-warmup)/max(1,total_steps-warmup)))
            group['lr'] = cfg['lrs'][method]*factor

def rng_state():
    return {'python':random.getstate(), 'numpy':np.random.get_state(), 'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}

def restore_rng(value):
    random.setstate(value['python']); np.random.set_state(value['numpy']); torch.set_rng_state(value['torch'])
    if value['cuda']: torch.cuda.set_rng_state_all(value['cuda'])

def save_checkpoint(path, model, optimizer, step, cfg, method, totals):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {'adapter':{n:t.detach().cpu().clone() for n,t in adapter_tensors(model).items()},
               'optimizer':optimizer.state_dict(), 'step':step, 'config':cfg, 'method':method,
               'rng':rng_state(), 'totals':totals, 'format_version':1}
    tmp = path.with_suffix('.tmp')
    torch.save(payload, tmp)
    if path.exists():
        # Retain the last complete checkpoint; all files live in group space.
        os.replace(path, path.with_name(path.stem+'.previous.pt'))
    os.replace(tmp, path)

def load_checkpoint(path, model, optimizer, cfg, method):
    state = torch.load(path, map_location='cpu', weights_only=False)
    assert state['config'] == cfg and state['method'] == method, 'Resume protocol mismatch'
    tensors = adapter_tensors(model)
    assert tensors.keys() == state['adapter'].keys(), 'Missing adapter buffers on resume'
    with torch.no_grad():
        for name, target in tensors.items(): target.copy_(state['adapter'][name])
    b = backend(model, method)
    if b: b._sync_sparse_requires_grad_with_masks()
    optimizer.load_state_dict(state['optimizer'])
    restore_rng(state['rng'])
    return state

def take_step(model, optimizer, rows, pad_id, cfg, method, step, total_steps, amp=True):
    b = backend(model, method)
    collecting = b is not None and b.should_collect_gradients(step)
    if b: b.enable_gradient_capture(collecting, mode='snip')
    optimizer.zero_grad(set_to_none=True)
    set_lrs(optimizer, step, total_steps, cfg, method)
    target_tokens = sum(row['supervised_tokens'] for row in rows)
    nll = 0.
    device = next(model.parameters()).device
    for start in range(0, len(rows), cfg['microbatch']):
        inputs = collate(rows[start:start+cfg['microbatch']], pad_id, device)
        count = int((inputs['labels'][:,1:] != -100).sum())
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            loss = model(**inputs).loss
        if not torch.isfinite(loss): raise RuntimeError('Non-finite train loss')
        (loss * count/target_tokens).backward()
        nll += float(loss.detach()) * count
        if collecting:
            captured = b.accumulate_gradient_statistics()
            assert captured['updated_tensors'] > 0, 'No SNIP gradients captured'
    norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg['clip_grad_norm'], error_if_nonfinite=True)
    optimizer.step()
    activation = None
    if b and b.should_generate_masks(step+1):
        activation = b.generate_sparse_masks()
        assert activation['selected_positions'] == cfg['k'], 'Exact K mismatch'
        optimizer.state.clear()
    if b: b.enable_gradient_capture(False)
    return nll/target_tokens, float(norm), activation

@torch.no_grad()
def evaluate_nll(model, ds, pad_id, limit=None):
    was_training = model.training
    model.eval()
    numerator = 0.; denominator = 0
    for i in range(len(ds) if limit is None else min(limit,len(ds))):
        row = ds[i]
        batch = collate([row], pad_id, next(model.parameters()).device)
        with torch.autocast('cuda', dtype=torch.bfloat16): loss = model(**batch).loss
        if not torch.isfinite(loss): raise RuntimeError('Non-finite validation loss')
        numerator += float(loss) * row['supervised_tokens']; denominator += row['supervised_tokens']
    model.train(was_training)
    return {'nll':numerator/denominator, 'nll_sum':numerator, 'supervised_tokens':denominator}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', required=True, choices=['lora','unilora','prolosa'])
    parser.add_argument('--config',required=True)
    parser.add_argument('--run-dir',required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    cfg = json.loads(Path(args.config).read_text())
    assert cfg['method']==args.method
    assert cfg['snip_reduction']=='max' and cfg['budget']==(cfg['expected_D'] if args.method=='lora' else cfg['expected_D']//4)
    if args.smoke:
        cfg['global_batch'] = cfg['microbatch'] = 1
        cfg['warmup_steps'] = cfg['score_steps'] = 2
    train = load_from_disk(str(PREPARED/'train'))
    val = load_from_disk(str(PREPARED/'val'))
    if args.smoke: train = train.select([i for i in range(min(len(train),1000)) if train[i]['length']<=512][:64])
    cfg['data_manifest_sha256'] = sha256(PREPARED/'manifest.json')
    per_epoch = math.ceil(len(train)/cfg['global_batch'])
    total_steps = args.max_steps or (12 if args.smoke else cfg['epochs']*per_epoch)
    cfg['total_steps'] = total_steps
    stage = 'search_smoke' if args.smoke else 'llama31_smol_r4'
    out = Path(args.run_dir)
    assert out.resolve().is_relative_to(STORAGE.resolve())
    if (out/'metrics.json').exists(): raise RuntimeError('Completed run already exists')
    if out.exists() and not args.resume: raise RuntimeError('Run directory exists; resume explicitly')
    out.mkdir(parents=True,exist_ok=True)
    tokenizer = load_tokenizer()
    model, D = build_model(args.method,cfg)
    assert D == cfg['expected_D']
    assert not model.get_output_embeddings().weight.requires_grad
    assert not model.get_input_embeddings().weight.requires_grad
    if args.method=='unilora': assert sum(p.numel() for p in model.parameters() if p.requires_grad)==cfg['budget']
    if args.method=='prolosa': assert sum(p.numel() for p in model.parameters() if p.requires_grad)==cfg['d']
    if args.method=='lora': assert sum(p.numel() for p in model.parameters() if p.requires_grad)==D
    model = model.to('cuda').train()
    optimizer = optimizer_for(model,cfg,args.method)
    manifest = {'config':cfg,'method':args.method,'stage':stage,'D':D,
        'gpu':torch.cuda.get_device_name(0),'job_id':os.environ.get('SLURM_JOB_ID'),
        'code_sha':subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip(),
        'script_sha256':sha256(__file__), 'peft_import':__import__('peft').__file__,
        'adapter_parameter_elements':sum(p.numel() for n,p in model.named_parameters() if 'lora_' in n),
        'adapter_tensor_bytes':sum(t.numel()*t.element_size() for t in adapter_tensors(model).values()),
        'effective_budget':D if args.method=='lora' else cfg['budget'],
        'started_at':time.time(),'initialization':str(model.peft_config['default'])}
    manifest['local_peft_source_sha256']=hashlib.sha256(''.join(
        sha256(p) for p in sorted((REPO/'math_instruction_tuning/peft/src/peft').rglob('*.py'))).encode()).hexdigest()
    manifest['common_script_sha256']=sha256(ROUND/'common.py')
    manifest['development_script_sha256']=sha256(HERE/'development.py')
    manifest['generation_checks_sha256']=sha256(ROUND/'generation_checks.py')
    manifest['environment_lock_sha256']=sha256(ROOT/'requirements.txt')
    if args.resume:
        previous=json.loads((out/'run_manifest.json').read_text())
        for key in ['config','script_sha256','local_peft_source_sha256','common_script_sha256','generation_checks_sha256','development_script_sha256','environment_lock_sha256']:
            assert previous[key]==manifest[key], f'Resume code/config changed: {key}'
        previous.setdefault('resume_jobs',[]).append(manifest['job_id'])
        write_json(out/'run_manifest.json',previous)
    else:
        write_json(out/'run_manifest.json',manifest)
    totals = {'examples':0,'nonpadding_tokens':0,'supervised_tokens':0,'step_seconds':[], 'resumes':[]}
    start_step = 0
    if args.resume:
        state=load_checkpoint(out/'checkpoint.pt',model,optimizer,cfg,args.method)
        start_step=state['step']; totals=state['totals']; totals['resumes'].append({'step':start_step,'job':os.environ.get('SLURM_JOB_ID')})
    if args.resume and start_step==100 and not args.smoke:
        check_generation(model,tokenizer,out,'step100',require_eos=True)
    stop_requested = [False]
    signal.signal(signal.SIGTERM,lambda *_:stop_requested.__setitem__(0,True))
    signal.signal(signal.SIGUSR1,lambda *_:stop_requested.__setitem__(0,True))
    epoch_orders = {}
    start_time=time.monotonic()
    with (out/'train.jsonl').open('a') as log:
        for step in range(start_step,total_steps):
            epoch, batch_index = divmod(step,per_epoch)
            if epoch not in epoch_orders:
                epoch_orders = {epoch:np.random.default_rng(cfg['seed']+1000003*epoch).permutation(len(train))}
            indices=epoch_orders[epoch][batch_index*cfg['global_batch']:(batch_index+1)*cfg['global_batch']]
            batch_rows=[train[int(i)] for i in indices]
            torch.cuda.synchronize(); tick=time.monotonic()
            loss,norm,activated=take_step(model,optimizer,batch_rows,tokenizer.pad_token_id,cfg,args.method,step,total_steps)
            torch.cuda.synchronize(); seconds=time.monotonic()-tick
            totals['step_seconds'].append(seconds)
            totals['examples']+=len(batch_rows)
            totals['nonpadding_tokens']+=sum(r['length'] for r in batch_rows)
            totals['supervised_tokens']+=sum(r['supervised_tokens'] for r in batch_rows)
            rec={'step':step+1,'loss':loss,'grad_norm':norm,'seconds':seconds,
                 'lrs':{g['kind']:g['lr'] for g in optimizer.param_groups},'activation':activated,
                 'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
                 'batch_row_hash':hashlib.sha256(','.join(r['row_id'] for r in batch_rows).encode()).hexdigest()}
            log.write(json.dumps(rec)+'\n'); log.flush()
            if step%10==0 or activated: print(json.dumps(rec),flush=True)
            if (step+1)%cfg['checkpoint_every']==0 or activated or step+1==cfg['warmup_steps'] or stop_requested[0]:
                tick=time.monotonic()
                save_checkpoint(out/'checkpoint.pt',model,optimizer,step+1,cfg,args.method,totals)
                print('CHECKPOINT',step+1,'seconds',time.monotonic()-tick,flush=True)
            if not args.smoke and step+1==100:
                write_json(out/'validation_step100.json',evaluate_nll(model,val,tokenizer.pad_token_id,limit=128))
                check_generation(model,tokenizer,out,'step100',require_eos=True)
            if stop_requested[0]: return
    save_checkpoint(out/'checkpoint.pt',model,optimizer,total_steps,cfg,args.method,totals)
    b=backend(model,args.method)
    sparse_stats=b.get_sparse_parameter_stats() if b else None
    if b:
        assert b.get_sparse_structure_stats()['selected_positions']==cfg['k']
        assert sparse_stats['selected_sparse_nonzero']>0
    train_seconds=time.monotonic()-start_time
    model.save_pretrained(str(out/'adapter'),safe_serialization=True)
    validation=evaluate_nll(model,val,tokenizer.pad_token_id,limit=16 if args.smoke else None)
    metrics={'validation':validation,'steps':total_steps,'train_seconds_this_session':train_seconds,
             'peak_gpu_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
             'peak_gpu_reserved_gib':torch.cuda.max_memory_reserved()/2**30,
             'max_host_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
             'tokens_per_second_steps_only':totals['nonpadding_tokens']/sum(totals['step_seconds']),
             'sparse_stats':sparse_stats,'totals':totals}
    metrics['generation_check']=check_generation(model,tokenizer,out,'smoke' if args.smoke else 'final',require_eos=not args.smoke)
    if not args.smoke:
        sys.path.insert(0,str(HERE))
        from development import evaluate as evaluate_dev
        metrics['development']=evaluate_dev(model,tokenizer,out)
    write_json(out/'metrics.json',metrics)
    print('RUN_COMPLETE',out,validation,flush=True)

if __name__=='__main__': main()
