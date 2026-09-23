"""Behavioral tests for the three actual tuners, including mid-scoring resume."""
import argparse
import copy
import json
import tempfile
from pathlib import Path
import torch
from peft import PeftModel
from transformers import LlamaConfig, LlamaForCausalLM
from common import CONFIG, STORAGE, write_json
from train import (adapter_tensors, backend, build_model, load_checkpoint,
                      optimizer_for, save_checkpoint, seed_all, take_step)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--cpu',action='store_true');args=parser.parse_args()
    device='cpu' if args.cpu else 'cuda'
    assert args.cpu or torch.cuda.is_available()
    torch.set_num_threads(2)
    cfg=copy.deepcopy(CONFIG)
    cfg.update(budget=128,d=96,k=32,warmup_steps=2,score_steps=2,microbatch=1,global_batch=2)
    rows=[{'input_ids':[1,2,3,4,5,6,7], 'labels':[-100,-100,3,4,5,6,7],
           'supervised_tokens':5,'row_id':'tiny:0','length':7},
          {'input_ids':[1,8,9,10,11], 'labels':[-100,-100,9,10,11],
           'supervised_tokens':3,'row_id':'tiny:1','length':5}]
    results={}
    for method in ['lora','unilora','prolosa']:
        m,D=build_model(method,cfg,tiny=True); m=m.to(device).train()
        m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        m.enable_input_require_grads()
        opt=optimizer_for(m,cfg,method)
        with tempfile.TemporaryDirectory(dir=STORAGE/'tmp',prefix=f'tiny_{method}_') as tmp:
            tmp=Path(tmp)
            for step in range(8):
                take_step(m,opt,rows,0,cfg,method,step,8,amp=False)
                if step+1 in (3,5): save_checkpoint(tmp/f'step{step+1}.pt',m,opt,step+1,cfg,method,{})
            expected={n:t.detach().cpu().clone() for n,t in adapter_tensors(m).items()}
            for resume_step in (3,5):
                resumed,_=build_model(method,cfg,tiny=True); resumed=resumed.to(device).train()
                resumed.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
                resumed.enable_input_require_grads()
                ropt=optimizer_for(resumed,cfg,method)
                state=load_checkpoint(tmp/f'step{resume_step}.pt',resumed,ropt,cfg,method)
                for step in range(state['step'],8): take_step(resumed,ropt,rows,0,cfg,method,step,8,amp=False)
                for name,t in adapter_tensors(resumed).items():
                    torch.testing.assert_close(t.detach().cpu(),expected[name],rtol=1e-5,atol=1e-7,msg=f'{method} resume {resume_step} {name}')
                del resumed,ropt
            m.eval()
            ids=torch.tensor([[1,2,3,4,5]],device=device)
            with torch.no_grad(): before=m(input_ids=ids).logits
            m.save_pretrained(str(tmp/'adapter'),safe_serialization=True)
            seed_all(cfg['seed'])
            base=LlamaForCausalLM(LlamaConfig(vocab_size=128,hidden_size=64,intermediate_size=128,
                num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=16,
                max_position_embeddings=256,attention_dropout=0.0))
            restored=PeftModel.from_pretrained(base,str(tmp/'adapter')).to(device).eval()
            with torch.no_grad(): after=restored(input_ids=ids).logits
            torch.testing.assert_close(after,before,rtol=1e-4,atol=1e-5,msg=f'{method} adapter reload')
            merged=restored.merge_and_unload().eval()
            with torch.no_grad(): merged_logits=merged(input_ids=ids).logits
            torch.testing.assert_close(merged_logits,before,rtol=2e-4,atol=2e-5,msg=f'{method} merged logits')
            if method=='prolosa':
                b=backend(m,method)
                assert b.get_sparse_structure_stats()['selected_positions']==cfg['k']
                assert b.get_sparse_parameter_stats()['selected_sparse_nonzero']>0
            results[method]={'D':D,'mid_scoring_resume':True,'post_activation_resume':True,
                             'adapter_reload':True,'merged_logits':True}
            print('TINY_TEST_PASSED',method,flush=True)
            del m,opt,restored,merged,base
            torch.cuda.empty_cache()
    write_json(STORAGE/'logs'/('round2_tiny_training_cpu_checks.json' if args.cpu else 'round2_tiny_training_checks.json'),results)

if __name__=='__main__': main()
