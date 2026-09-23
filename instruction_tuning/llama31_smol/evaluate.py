"""Frozen greedy IFEval/IFBench generation with official strict/loose scorers."""
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
import argparse
import dataclasses
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_from_disk
from langdetect import DetectorFactory
from common import REPO, CONFIG, ROOT, STORAGE, MODEL, PREPARED, sha256, write_json, serialize, load_tokenizer, ROUND
from train import evaluate_nll
assert Path(__import__('peft').__file__).resolve().is_relative_to((REPO/'math_instruction_tuning/peft/src').resolve()), 'Expected frozen original max PEFT'

def scorers():
    DetectorFactory.seed=0
    sys.path.insert(0,str(STORAGE/'external/google_ifeval'))
    from instruction_following_eval import evaluation_lib as google
    path=STORAGE/'external/IFBench/evaluation_lib.py'
    spec=importlib.util.spec_from_file_location('i1_ifbench_evaluation_lib',path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module; spec.loader.exec_module(module)
    return {'ifeval':google,'ifbench':module}

def score_predictions(name, lib, predictions):
    records=[json.loads(l) for l in (STORAGE/f'datasets/benchmarks/{name}.jsonl').read_text().splitlines()]
    by_prompt={x['prompt']:x['response'] for x in predictions}
    assert len(predictions)==len(records) and set(by_prompt)=={x['prompt'] for x in records}
    result={}; detailed=[]
    for mode in ('strict','loose'):
        outputs=[]
        random.seed(101)
        for row in records:
            inp=lib.InputExample(key=row['key'],instruction_id_list=row['instruction_id_list'],prompt=row['prompt'],kwargs=row['kwargs'])
            # Exceptions propagate: infra/verifier failures are not silently scored as 0.
            output=getattr(lib,f'test_instruction_following_{mode}')(inp,by_prompt)
            outputs.append(output)
            detailed.append({'key':row['key'],'mode':mode,**dataclasses.asdict(output)})
        result[f'prompt_{mode}']=sum(x.follow_all_instructions for x in outputs)/len(outputs)
        result[f'instruction_{mode}']=sum(sum(x.follow_instruction_list) for x in outputs)/sum(len(x.follow_instruction_list) for x in outputs)
    return result,detailed

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--method',choices=['lora','unilora','prolosa','base'],required=True)
    parser.add_argument('--config',required=True)
    parser.add_argument('--run-dir',required=True)
    args=parser.parse_args()
    CONFIG=json.loads(Path(args.config).read_text())
    assert CONFIG['method']==args.method
    assert torch.cuda.is_available()
    libs=scorers()
    out=Path(args.run_dir)
    assert out.resolve().is_relative_to(STORAGE.resolve())
    if args.method!='base': assert (out/'metrics.json').exists(), 'Training not complete'
    results=out/'evaluation'; results.mkdir(parents=True,exist_ok=True)
    tok=load_tokenizer()
    provenance={'model_id':CONFIG['model_id'],'model_revision':CONFIG['model_revision'],
        'method':args.method,'eval_script_sha256':sha256(__file__),
        'data_manifest_sha256':sha256(PREPARED/'manifest.json'),
        'serializer_sha256':sha256(ROUND/'common.py'),
        'config_sha256':sha256(args.config),
        'environment_lock_sha256':sha256(ROOT/'requirements.txt'),
        'verifier_sources':json.loads((STORAGE/'manifests/verifier_sources.json').read_text()),
        'langdetect_seed':0,'python_scoring_seed':101,
        'weights_mode':('BF16 base; no adapter' if args.method=='base' else 'BF16 base with active FP32 adapter; not merged'),
        'adapter_files':({p.name:sha256(p) for p in sorted((out/'adapter').iterdir()) if p.is_file()}
                         if args.method!='base' else {})}
    manifest=results/'protocol.json'
    if manifest.exists():
        assert json.loads(manifest.read_text())==provenance, 'Evaluation resume provenance mismatch'
    else:
        write_json(manifest,provenance)
    model=AutoModelForCausalLM.from_pretrained(str(MODEL),local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa')
    if args.method!='base':
        model=PeftModel.from_pretrained(model,str(out/'adapter'))
    model=model.cuda().eval()
    model.gradient_checkpointing_disable()
    model.config.use_cache=True
    stops=[tok.eos_token_id]
    max_new=CONFIG['generation_max_new_tokens']
    summary={'method':args.method,'job':os.environ.get('SLURM_JOB_ID'), 'decoding':{
        'do_sample':False,'max_new_tokens':max_new,'stop_ids':stops,'serializer':'plain_role_headers_native_eos_v1',
        'initial_context':8192,'overflow_rule':'expand to exact prompt+max_new_tokens within base max_position_embeddings'},
        'benchmarks':{}}
    for name,lib in libs.items():
        records=[json.loads(l) for l in (STORAGE/f'datasets/benchmarks/{name}.jsonl').read_text().splitlines()]
        file=results/f'{name}_predictions.jsonl'
        predictions=[json.loads(l) for l in file.read_text().splitlines()] if file.exists() else []
        assert [x['prompt'] for x in predictions]==[x['prompt'] for x in records[:len(predictions)]]
        with file.open('a') as handle:
            for i,row in enumerate(records[len(predictions):],start=len(predictions)):
                prompt_ids,_=serialize([{'role':'user','content':row['prompt']}],tok,True)
                ids=torch.tensor([prompt_ids],device='cuda')
                requested=ids.shape[1]+max_new
                assert requested<=model.config.max_position_embeddings, 'Prompt exceeds model context; do not drop it'
                tick=time.monotonic()
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                    generated=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),max_new_tokens=max_new,
                        do_sample=False,temperature=None,top_p=None,top_k=None,eos_token_id=stops,pad_token_id=tok.pad_token_id)
                new=generated[0,ids.shape[1]:].tolist()
                stopped=bool(new and new[-1] in stops)
                answer_ids=new[:-1] if stopped else new
                pred={'key':row['key'],'prompt':row['prompt'],'response':tok.decode(answer_ids,skip_special_tokens=False),
                    'prompt_tokens':ids.shape[1],'generated_tokens':len(new),'stop_reason':'eos' if stopped else 'length',
                    'context_budget':max(8192,requested),'seconds':time.monotonic()-tick}
                handle.write(json.dumps(pred,ensure_ascii=False)+'\n');handle.flush();predictions.append(pred)
                if i%20==0: print('GENERATED',name,i+1,'of',len(records),flush=True)
        metrics,detail=score_predictions(name,lib,predictions)
        with (results/f'{name}_scores.jsonl').open('w') as handle:
            for row in detail:handle.write(json.dumps(row,ensure_ascii=False)+'\n')
        metrics.update(n=len(predictions),empty=sum(not p['response'].strip() for p in predictions),
            length_truncated=sum(p['stop_reason']=='length' for p in predictions),predictions_sha256=sha256(file))
        summary['benchmarks'][name]=metrics
        write_json(results/'metrics_partial.json',summary)
    model.config.use_cache=False
    summary['sft_test']=evaluate_nll(model,load_from_disk(str(PREPARED/'test')),tok.pad_token_id)
    write_json(results/'metrics.json',summary)
    print('EVALUATION_COMPLETE',json.dumps(summary),flush=True)

if __name__=='__main__':main()
