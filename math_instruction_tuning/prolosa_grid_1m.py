"""Plan, execute and summarize a bounded r=4, B=2**20 ProLoSA grid.

Primary score is the unweighted GSM8K/MATH500 test accuracy mean. This is an
exploratory test-set search, not validation-selected or multi-seed evidence.
"""
import argparse
import itertools
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PLAN = ROOT / 'prolosa_grid_1m.json'
RESULTS = ROOT / 'results/prolosa_grid_1m'
BASE = 'NousResearch/Meta-Llama-3.1-8B'
BUDGET = 1048576


def config(ratio=12, td_lr=0.0016, warmup=128, mask_steps=1,
           sparse_mult=0.2, init_bound=0.02, reset=True):
    # Integer round-to-nearest; always allocate the remainder to d.
    sparse = int(BUDGET / (ratio + 1) + 0.5)
    return dict(ratio=ratio, d=BUDGET-sparse, K=sparse, td_lr=td_lr,
                warmup=warmup, mask_steps=mask_steps, sparse_mult=sparse_mult,
                init_bound=init_bound, reset=reset, rank=4, seed=42,
                budget=BUDGET, base_lr=0.0002, scheduler_warmup_ratio=0.02,
                sparse_decay=True)


def key(c):
    return json.dumps(c, sort_keys=True)


def plan():
    old=[]
    for rate,tag in [(0.0016,'1.6e-3'),(0.0032,'3.2e-3')]:
        for ratio in (2,4,8,12):
            c=config(ratio=ratio,td_lr=rate)
            cdir=ROOT/f'results/p0_math_prolosa1M/td{tag}/llama31_prolosa_r{ratio}to1_s42'
            # Reuse only completed evaluations.
            for benchmark in ('gsm8k','math500'):
                read_score(cdir/ (benchmark+'.log'), 1319 if benchmark=='gsm8k' else 500)
            old.append(dict(config=c,result_dir=str(cdir.relative_to(ROOT)),source='existing'))
    trials=[]
    def add(c,section):
        if key(c) in {key(x['config']) for x in old+trials}:
            return
        trials.append(dict(config=c,section=section))
    # Start with a delayed, more stable mask estimate.
    add(config(warmup=512,mask_steps=8),'timing_sparse_lr')
    for w,m,s in itertools.product((128,512,1024),(1,8),(0.1,0.2)):
        add(config(warmup=w,mask_steps=m,sparse_mult=s),'timing_sparse_lr')
    # Local grid at a fixed delayed-start anchor, NOT selected using future scores.
    for ratio,rate in itertools.product((4,8,12,24),(0.0008,0.0016,0.0024)):
        add(config(ratio=ratio,td_lr=rate,warmup=512,mask_steps=8),'ratio_projected_lr')
    for bound in (0.01,0.04):
        add(config(warmup=512,mask_steps=8,init_bound=bound),'initialization')
    add(config(warmup=512,mask_steps=8,reset=False),'optimizer_reset')
    assert len(trials)==25
    for i,entry in enumerate(trials):
        entry.update(task_id=i,run_id=f'grid_{i:02d}',result_dir=f'results/prolosa_grid_1m/grid_{i:02d}')
        c=entry['config']; assert c['d']+c['K']==BUDGET and c['rank']==4
    document=dict(description=__doc__,existing=old,trials=trials)
    if PLAN.exists():
        assert json.loads(PLAN.read_text())==document, 'Refusing to replace a different submitted grid'
    else:
        PLAN.write_text(json.dumps(document,indent=2)+'\n')
    print(f'Plan: {len(old)} existing + {len(trials)} new configurations; {PLAN}')


def read_score(path, count):
    text=Path(path).read_text(errors='replace')
    values=re.findall(r'length====\s*(\d+)\s*,\s*(?:gsm8k )?acc====\s*([0-9.]+)',text)
    if not values or int(values[-1][0])!=count:
        raise ValueError(f'Incomplete/incorrect-sized evaluation: {path}')
    return 100*float(values[-1][1])


def execute(command,log):
    print('Running:', ' '.join(map(str,command)), '\nLog:', log, flush=True)
    with log.open('w') as stream:
        subprocess.run(list(map(str,command)),cwd=ROOT,stdout=stream,
                       stderr=subprocess.STDOUT,check=True)


def run(task_id):
    import os
    import torch
    if torch.cuda.device_count()!=1:
        raise RuntimeError('Exactly one GPU must be visible')
    entry=json.loads(PLAN.read_text())['trials'][task_id]
    c=entry['config']
    assert c['d']+c['K']==BUDGET and c['rank']==4
    dest=ROOT/entry['result_dir']
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError(f'Refusing to overwrite existing run: {dest}')
    dest.mkdir(parents=True)
    output=ROOT/'output/prolosa_grid_1m'/entry['run_id']
    merged=ROOT/'output_merged/prolosa_grid_1m'/entry['run_id']
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    manifest=dict(entry,slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                  gpu=torch.cuda.get_device_name(0),gpu_count=1,
                  sparse_activation_step=c['warmup']+c['mask_steps'])
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    train=[sys.executable,'intruction_tuning_unilora_multi_gpu.py',
           '--model_name_or_path',BASE,'--output_dir',output,
           '--unilora_variant','unilora_rosa_snip','--lora_r','4',
           '--theta_d_length',c['d'],'--rosa_sparse_budget',c['K'],
           '--init_theta_d_bound',c['init_bound'],
           '--learning_rate',c['base_lr'],'--learning_rate_theta_d',c['td_lr'],
           '--learning_rate_vector_bank',c['td_lr'],
           '--rosa_sparse_lr_mult',c['sparse_mult'],
           '--rosa_warmup_steps',c['warmup'],'--rosa_mask_steps',c['mask_steps'],
           '--rosa_reset_optimizer_on_mask',str(c['reset']),
           '--rosa_decay_sparse_lr_after_activation','True',
           '--data_path','meta-math/MetaMathQA','--dataset_split','train[:100000]',
           '--dataset_field','query','response','--model_max_length','512',
           '--num_train_epochs','2','--per_device_train_batch_size','1',
           '--gradient_accumulation_steps','64','--gradient_checkpointing','True',
           '--save_strategy','steps','--save_steps','100','--save_total_limit','2',
           '--weight_decay','0','--warmup_ratio','0.02','--lr_scheduler_type','cosine',
           '--logging_steps','10','--bf16','True','--tf32','True','--fp16','False',
           '--device_map','auto','--max_memory_per_gpu','44GiB',
           '--max_memory_cpu','64GiB','--report_to','tensorboard','--seed','42']
    (dest/'train_command.json').write_text(json.dumps(list(map(str,train)),indent=2)+'\n')
    try:
        execute(train,dest/'train.log')
        adapters=list(output.glob('**/ft/adapter_config.json'))
        if len(adapters)!=1:
            raise RuntimeError(f'Expected exactly one final adapter: {adapters}')
        adapter=adapters[0].parent
        saved=json.loads(adapters[0].read_text())
        assert saved['r']==4 and saved['theta_d_length']==c['d']
        state=json.loads((adapter.parent/'trainer_state.json').read_text())
        assert state['epoch']>=1.99 and state['global_step']==state['max_steps']
        text=(dest/'train.log').read_text(errors='replace')
        activation=re.search(r'Activated UniLoRA-RoSA-SNIP sparse compensation:.*selected_positions=(\d+)',text)
        assert activation and int(activation[1])==c['K'], 'Sparse budget/activation mismatch'
        manifest.update(adapter_path=str(adapter),completed_steps=state['global_step'],
                        actual_selected_sparse_positions=int(activation[1]),
                        actual_active_trainable_params=c['d']+int(activation[1]))
        (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        execute([sys.executable,'-m','utils.merge_adapter_to_base_model','--base_model',BASE,
                 '--adapter',adapter,'--output_path',merged,'--dtype','bfloat16'],dest/'merge.log')
        for bench,file,script in [('gsm8k','gsm8k_test.jsonl','gsm8k_eval.py'),
                                  ('math500','MATH500_test.jsonl','MATH_eval.py')]:
            execute([sys.executable,f'instruction_tuning_eval/{script}','--model',merged,
                     '--data_file',f'data/math_eval/{file}','--batch_size','32',
                     '--tensor_parallel_size','1','--max_model_len','4096'],dest/f'{bench}.log')
        gsm=read_score(dest/'gsm8k.log',1319); math=read_score(dest/'math500.log',500)
        scores=dict(gsm8k=gsm,math500=math,mean=(gsm+math)/2)
        (dest/'scores.json').write_text(json.dumps(scores,indent=2)+'\n')
        (dest/'COMPLETE').write_text('Training, budget checks and both evaluations succeeded.\n')
        print('COMPLETE',entry['run_id'],scores,flush=True)
    except Exception as error:
        (dest/'FAILED').write_text(f'{type(error).__name__}: {error}\n')
        raise


def summarize():
    doc=json.loads(PLAN.read_text()); rows=[]; status=[]
    for entry in doc['existing']+doc['trials']:
        dest=ROOT/entry['result_dir']; name=entry.get('run_id',entry['result_dir'])
        try:
            if entry.get('source')!='existing' and not (dest/'COMPLETE').exists():
                raise ValueError('FAILED' if (dest/'FAILED').exists() else 'pending/incomplete')
            g=read_score(dest/'gsm8k.log',1319); m=read_score(dest/'math500.log',500)
            rows.append(dict(entry,gsm8k=g,math500=m,mean=(g+m)/2))
        except (OSError,ValueError) as error:
            status.append(f'- {name}: {error}')
    rows.sort(key=lambda r:r['mean'],reverse=True)
    lines=['# ProLoSA r=4, B=1,048,576: exploratory grid results','',
           'Seed 42; GSM8K and MATH500 exact match (%). Primary ranking is their unweighted mean.',
           'Selection uses these test scores; it is not independent validation or multi-seed evidence.',
           f'Completed: {len(rows)} / {len(doc["existing"])+len(doc["trials"])} configurations.','']
    if rows:
        for metric in ('mean','gsm8k','math500'):
            best=max(rows,key=lambda r:r[metric])
            lines.append(f'Best {metric}: {best[metric]:.4f}; {best.get("run_id",best["result_dir"])}; GSM8K={best["gsm8k"]:.2f}, MATH500={best["math500"]:.2f}.')
    lines+=['','| Run | d:K | d | K | theta LR | sparse multiplier | Warmup | Mask window | Init | Reset optimizer | GSM8K | MATH500 | Mean |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|']
    for row in rows:
        c=row['config']; name=row.get('run_id',row['result_dir'])
        lines.append(f'| {name} | {c["ratio"]}:1 | {c["d"]} | {c["K"]} | {c["td_lr"]} | {c["sparse_mult"]} | {c["warmup"]} | {c["mask_steps"]} | {c["init_bound"]} | {c["reset"]} | {row["gsm8k"]:.2f} | {row["math500"]:.2f} | {row["mean"]:.4f} |')
    lines+=['','Incomplete runs:',*status]
    RESULTS.mkdir(parents=True,exist_ok=True)
    (RESULTS/'summary.md').write_text('\n'.join(lines)+'\n')
    (RESULTS/'summary.json').write_text(json.dumps(dict(completed=rows,incomplete=status),indent=2)+'\n')
    print('\n'.join(lines[:11]))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['plan','run','summarize'])
    parser.add_argument('--task-id',type=int)
    args=parser.parse_args()
    if args.action=='plan': plan()
    elif args.action=='run': run(args.task_id)
    else: summarize()
