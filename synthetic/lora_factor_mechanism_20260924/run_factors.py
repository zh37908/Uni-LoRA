"""Actual BA LoRA and source-derived normalized sharing, with locked dev tuning."""
import argparse
import ast
import json
import math
import os
from pathlib import Path
import sys
import time
import types
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'predictive_bias_variance_20260924'
sys.path.insert(0,str(OLD))
from run import base,sha,atomic,TEACHER
METHODS=['lora_matched','unilora','lora_zero']
SHAPES=[(4,64),(64,4),(4,64),(64,4)]


def source_method(path,cls,name):
    tree=ast.parse(path.read_text())
    klass=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls)
    function=next(n for n in klass.body if isinstance(n,ast.FunctionDef) and n.name==name)
    function.decorator_list=[];function.returns=None
    for arg in function.args.args+function.args.kwonlyargs:
        arg.annotation=None
    module=ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[]))
    space={'torch':torch,'F':F}
    exec(compile(module,str(path),'exec'),space)
    return space[name]


RECONSTRUCT=source_method(HERE/'unilora_layer_source.py','Linear','_get_lora_matrices')
SOURCE_FORWARD=source_method(HERE/'unilora_layer_source.py','Linear','forward')
GENERATE=source_method(HERE/'unilora_model_source.py','UniLoRAModel','generate_index')


def setup():
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
    torch.set_num_threads(4);torch.manual_seed(6000);torch.cuda.manual_seed_all(6000)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False


def model_load():
    payload=torch.load(TEACHER,map_location='cpu',weights_only=False)
    model=base.Block().cuda();model.load_state_dict(payload['state']);model.eval().requires_grad_(False)
    return model


class Factors(nn.Module):
    def __init__(self,method,bundle):
        super().__init__();self.method=method
        if method=='unilora':
            self.bank=nn.Parameter(bundle['bank'].cuda().clone())
            self.register_buffer('assignment',bundle['assignment'].cuda())
            self.register_buffer('scales',bundle['scales'].cuda())
        else:
            values=bundle['matched'] if method=='lora_matched' else bundle['standard']
            self.params=nn.ParameterList([nn.Parameter(t.cuda().clone()) for t in values])
    def matrices(self):
        if self.method!='unilora':return list(self.params)
        matrices=[]
        for layer in range(2):
            start=512*layer
            holder=types.SimpleNamespace(unilora_theta_d={'default':self.bank},
                unilora_indices_A={'default':self.assignment[start:start+256].reshape(4,64)},
                unilora_indices_B={'default':self.assignment[start+256:start+512].reshape(64,4)},
                unilora_scales_A={'default':self.scales[start:start+256].reshape(4,64)},
                unilora_scales_B={'default':self.scales[start+256:start+512].reshape(64,4)})
            matrices.extend(RECONSTRUCT(holder,'default'))
        return matrices
    def forward(self):
        qA,qB,vA,vB=self.matrices()
        return torch.cat([(qB@qA).flatten(),(vB@vA).flatten()])
    def vector(self):return torch.cat([t.flatten() for t in self.matrices()])


def source_audit(bundle):
    adapter=Factors('unilora',bundle);matched=Factors('lora_matched',bundle)
    torch.testing.assert_close(adapter.vector(),matched.vector(),rtol=0,atol=0)
    torch.testing.assert_close(adapter(),matched(),rtol=0,atol=0)
    g=torch.Generator(device='cuda').manual_seed(970099)
    x=torch.randn(3,5,64,generator=g,device='cuda')
    layer=nn.Linear(64,64,bias=False).cuda().requires_grad_(False)
    holder=types.SimpleNamespace(unilora_theta_d={'default':adapter.bank},
        unilora_indices_A={'default':adapter.assignment[:256].reshape(4,64)},
        unilora_indices_B={'default':adapter.assignment[256:512].reshape(64,4)},
        unilora_scales_A={'default':adapter.scales[:256].reshape(4,64)},
        unilora_scales_B={'default':adapter.scales[256:512].reshape(64,4)},
        disable_adapters=False,merged=False,base_layer=layer,active_adapters=['default'],
        unilora_dropout={'default':nn.Identity()})
    holder._get_lora_matrices=types.MethodType(RECONSTRUCT,holder)
    actual=SOURCE_FORWARD(holder,x)
    ours=F.linear(x,layer.weight+adapter()[:4096].reshape(64,64))
    output_error=float((actual-ours).abs().max())
    probe=torch.randn(actual.shape,generator=g,device='cuda')
    grad_actual=torch.autograd.grad((actual*probe).sum(),adapter.bank,retain_graph=True)[0]
    grad_ours=torch.autograd.grad((ours*probe).sum(),adapter.bank)[0]
    grad_error=float((grad_actual-grad_ours).norm()/grad_actual.norm())
    assert output_error<2e-6 and grad_error<2e-5
    assert sum(p.numel() for p in adapter.parameters())==128
    assert sum(p.numel() for p in matched.parameters())==1024
    return dict(source_forward_max_error=output_error,source_bank_gradient_relative_error=grad_error,
        matched_initial_factors_identical=True,matched_initial_update_identical=True,
        compressed_parameters=128,full_parameters=1024,rank=4,BA_scaling=1.)


def prepare():
    assert not (HERE/'prepared.json').exists()
    model=model_load()
    assignment=GENERATE(None,1024,128,42)
    scales=1/torch.bincount(assignment,minlength=128).float().sqrt()[assignment]
    bank=torch.empty(128,device='cuda').uniform_(-.02,.02,generator=torch.Generator(device='cuda').manual_seed(970010))
    initial=bank[assignment.cuda()]*scales.cuda()
    matched=[initial[i*256:(i+1)*256].reshape(shape).cpu() for i,shape in enumerate(SHAPES)]
    standard=[];g=torch.Generator(device='cuda').manual_seed(970011)
    for shape in SHAPES:
        t=torch.empty(shape,device='cuda')
        if shape==(4,64):nn.init.kaiming_uniform_(t,a=math.sqrt(5),generator=g)
        else:t.zero_()
        standard.append(t.cpu())
    bundle=dict(bank=bank.cpu(),assignment=assignment,scales=scales,matched=matched,standard=standard)
    audit=source_audit(bundle)
    g=torch.Generator(device='cuda').manual_seed(970020)
    tfactors=[torch.randn(shape,generator=g,device='cuda')/math.sqrt(shape[1]) for shape in SHAPES]
    def delta(scale):
        a,b,c,d=[t*math.sqrt(scale) for t in tfactors]
        return torch.cat([(b@a).flatten(),(d@c).flatten()])
    xcal=torch.randn(1024,32,64,generator=torch.Generator(device='cuda').manual_seed(970030),device='cuda')
    fcal=base.predict(model,xcal,None);target=.1*float(fcal.var(unbiased=False))
    def displacement(scale):return float((base.predict(model,xcal,delta(scale))-fcal).square().mean())
    lo,hi=0.,1.
    while displacement(hi)<target:
        hi*=2;assert hi<1e5
    for _ in range(30):
        mid=(lo+hi)/2
        if displacement(mid)<target:lo=mid
        else:hi=mid
    scale=(lo+hi)/2;teacher=delta(scale)
    actual=displacement(scale);assert abs(actual/target-1)<2e-5
    bundle.update(teacher=teacher.cpu(),teacher_factors=[(t*math.sqrt(scale)).cpu() for t in tfactors],
        calibration_inputs=xcal.cpu(),calibration_target=target,displacement=actual,noise_unit=math.sqrt(actual))
    xdev=torch.randn(1024,32,64,generator=torch.Generator(device='cuda').manual_seed(970040),device='cuda')
    bundle.update(dev_x=xdev.cpu(),dev_truth=base.predict(model,xdev,teacher).cpu())
    init_preds={m:base.predict(model,xdev,Factors(m,bundle)()).cpu() for m in METHODS}
    torch.testing.assert_close(init_preds['lora_matched'],init_preds['unilora'],rtol=0,atol=0)
    audit['standard_vs_matched_initial_prediction_mse']=float((init_preds['lora_zero']-init_preds['lora_matched']).square().mean())
    torch.save(bundle,HERE/'prepared.pt')
    dev_grid=[dict(method=m,lr=lr,noise_multiple=s,repeat=r) for m in METHODS for lr in [.001,.005,.02] for s in [0.,1.,4.] for r in range(3)]
    main_grid=[dict(method=m,noise_multiple=s,repeat=r,long_control=s in [1.,4.] and r<5)
               for s in [0.,.5,1.,2.,4.] for m in METHODS for r in range(20)]
    record=dict(source_sha256=sha(__file__),plan_sha256=sha(HERE/'PLAN.md'),bundle_sha256=sha(HERE/'prepared.pt'),
        teacher_base_sha256=sha(TEACHER),base_code_sha256=sha(base.__file__),
        repository_model_sha256=sha(HERE/'unilora_model_source.py'),repository_layer_sha256=sha(HERE/'unilora_layer_source.py'),
        target_id=base.digest(teacher.cpu().double().numpy()),dev_grid=dev_grid,main_grid=main_grid,
        calibration_displacement=actual,calibration_base_variance=target/.1,noise_unit=math.sqrt(actual),
        teacher_scale=scale,teacher_parameterization='rank4 BA on Q/V; realizable in full LoRA',audit=audit)
    atomic(HERE/'prepared.json',record);print('Prepared true factor experiment',audit,flush=True)


def require():
    record=json.loads((HERE/'prepared.json').read_text())
    for key,path in [('source_sha256',Path(__file__)),('plan_sha256',HERE/'PLAN.md'),('bundle_sha256',HERE/'prepared.pt'),
        ('teacher_base_sha256',TEACHER),('base_code_sha256',Path(base.__file__)),
        ('repository_model_sha256',HERE/'unilora_model_source.py'),('repository_layer_sha256',HERE/'unilora_layer_source.py')]:
        assert record[key]==sha(path),key
    return record,torch.load(HERE/'prepared.pt',map_location='cpu',weights_only=False)


def train(model,adapter,x,y,clean,xe,truth,lr,checkpoints):
    opt=torch.optim.AdamW(adapter.parameters(),lr=lr,weight_decay=0.)
    g=torch.Generator(device='cuda').manual_seed(81000)
    preds,factors,rows={},{},[];torch.cuda.synchronize();started=time.perf_counter()
    for step in range(1,max(checkpoints)+1):
        ix=torch.randint(256,(128,),generator=g,device='cuda')
        loss=F.mse_loss(model(x[ix],adapter()),y[ix]);opt.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(),1.);opt.step()
        if step in checkpoints:
            torch.cuda.synchronize();seconds=time.perf_counter()-started
            update=adapter().detach();pred=base.predict(model,xe,update).cpu().numpy()
            training=base.predict(model,x,update)
            assert np.isfinite(pred).all()
            preds[f'step{step}']=pred;factors[f'step{step}']=adapter.vector().detach().cpu().numpy()
            rows.append(dict(step=step,risk=float(np.mean((pred.astype(float)-truth)**2)),
                noisy_train_mse=float(F.mse_loss(training,y)),clean_train_mse=float(F.mse_loss(training,clean)),seconds=seconds))
    return preds,factors,rows


def run(mode,shard,shards):
    record,bundle=require();model=model_load();teacher=bundle['teacher'].cuda()
    if mode=='main':
        selection=json.loads((HERE/'selection.json').read_text())
        assert selection['prepared_sha256']==sha(HERE/'prepared.json')
        tests=torch.load(HERE/'test.pt',map_location='cpu',weights_only=False)
        assert selection['test_sha256']==sha(HERE/'test.pt')
        xe=tests['x'].cuda();truth=tests['truth'].numpy().astype(float)
    else:xe=bundle['dev_x'].cuda();truth=bundle['dev_truth'].numpy().astype(float)
    grid=record['main_grid' if mode=='main' else 'dev_grid']
    for index in range(shard,len(grid),shards):
        spec=grid[index];out=HERE/mode/f'{index:04d}';out.mkdir(parents=True,exist_ok=True)
        if (out/'result.json').exists():
            old=json.loads((out/'result.json').read_text())
            assert old['source_sha256']==sha(__file__) and old['spec']==spec
            assert old['predictions_sha256']==sha(out/'predictions.npz')
            continue
        lr=selection['learning_rates'][spec['method']] if mode=='main' else spec['lr']
        g=torch.Generator(device='cuda').manual_seed((972000 if mode=='main' else 971000)+spec['repeat'])
        x=torch.randn(256,32,64,generator=g,device='cuda');noise=torch.randn(256,16,generator=g,device='cuda')
        clean=base.predict(model,x,teacher);sigma=spec['noise_multiple']*record['noise_unit'];y=clean+sigma*noise
        adapter=Factors(spec['method'],bundle)
        checkpoints=[1000,3000] if mode=='main' and spec['long_control'] else [1000]
        preds,factors,rows=train(model,adapter,x,y,clean,xe,truth,lr,checkpoints)
        np.savez_compressed(out/'predictions.npz',**preds);np.savez_compressed(out/'factors.npz',**factors)
        result=dict(spec=spec,mode=mode,lr=lr,sigma=sigma,rows=rows,source_sha256=sha(__file__),
            prepared_sha256=sha(HERE/'prepared.json'),target_id=record['target_id'],test_id=base.digest(xe.cpu().numpy()),
            train_x_id=base.digest(x.cpu().numpy()),noise_id=base.digest(noise.cpu().numpy()),train_y_id=base.digest(y.cpu().numpy()),
            predictions_sha256=sha(out/'predictions.npz'),factors_sha256=sha(out/'factors.npz'),
            gpu=torch.cuda.get_device_name(0),torch_version=torch.__version__,job=os.environ['SLURM_JOB_ID'])
        if mode=='main':result['selection_sha256']=sha(HERE/'selection.json')
        atomic(out/'result.json',result);print(mode,index,spec,lr,rows,flush=True)


def lock():
    record,bundle=require();assert not (HERE/'selection.json').exists()
    dev=[]
    for i,spec in enumerate(record['dev_grid']):
        p=HERE/'dev'/f'{i:04d}'/'result.json';row=json.loads(p.read_text())
        assert row['source_sha256']==sha(__file__) and row['spec']==spec
        dev.append(row)
    scores=[];selected={}
    for method in METHODS:
        for lr in [.001,.005,.02]:
            rows=[r for r in dev if r['spec']['method']==method and r['lr']==lr];assert len(rows)==9
            scores.append(dict(method=method,lr=lr,mean_dev_mse=float(np.mean([r['rows'][0]['risk'] for r in rows]))))
        selected[method]=min([r for r in scores if r['method']==method],key=lambda r:(r['mean_dev_mse'],r['lr']))['lr']
    # Selection is recorded before generating the final test stream.
    atomic(HERE/'dev_selection_before_test.json',dict(learning_rates=selected,scores=scores,prepared_sha256=sha(HERE/'prepared.json')))
    model=model_load();x=torch.randn(1024,32,64,generator=torch.Generator(device='cuda').manual_seed(970050),device='cuda')
    truth=base.predict(model,x,bundle['teacher'].cuda()).cpu()
    torch.save(dict(x=x.cpu(),truth=truth),HERE/'test.pt');np.save(HERE/'truth.npy',truth.numpy())
    atomic(HERE/'selection.json',dict(learning_rates=selected,scores=scores,prepared_sha256=sha(HERE/'prepared.json'),
        dev_selection_sha256=sha(HERE/'dev_selection_before_test.json'),test_sha256=sha(HERE/'test.pt'),truth_sha256=sha(HERE/'truth.npy'),
        test_id=base.digest(x.cpu().numpy()),dev_runs=81,dev_results_sha256={f'{i:04d}':sha(HERE/'dev'/f'{i:04d}'/'result.json') for i in range(81)}))
    print('LOCKED learning rates',selected,'before final evaluations',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','dev','lock','main']);p.add_argument('--shards',type=int,default=4)
    args=p.parse_args();setup()
    if args.mode=='prepare':prepare()
    elif args.mode=='lock':lock()
    else:run(args.mode,int(os.environ.get('SLURM_ARRAY_TASK_ID','0')),args.shards)
