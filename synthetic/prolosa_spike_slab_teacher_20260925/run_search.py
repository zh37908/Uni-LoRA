import os,sys,json,time,argparse
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
HERE=Path(__file__).resolve().parent;PRIOR=HERE.parent/'prolosa_all_linear_20260924'
sys.path.insert(0,str(PRIOR))
import run_all_linear as core
sha,atomic,base=core.sha,core.atomic,core.base
METHODS=['LoRA','Uni-LoRA','ProLoSA'];MULT=[0.,.5,1.,2.,4.];LRS=[.0002,.0005,.001]

def configs():
    rr=[]
    for method in METHODS[:2]:
        for lr in LRS:rr.append(dict(method=method,bank=0 if method=='LoRA' else 576,sparse=0,lr=lr,residual_lr_multiplier=1.,warmup=128,score_steps=128,ratio='full' if method=='LoRA' else 'bank only'))
    for d,k,ratio in [(288,288,'1:1'),(461,115,'~4:1'),(512,64,'8:1'),(542,34,'~16:1')]:
        for lr in LRS:
            for mult in [.25,1.,4.]:
                for warmup in [64,128,256]:rr.append(dict(method='ProLoSA',bank=d,sparse=k,lr=lr,residual_lr_multiplier=mult,warmup=warmup,score_steps=128,ratio=ratio))
    return {f'c{i:03d}':r for i,r in enumerate(rr)}

class Adapter(core.Adapter):
    def __init__(self,c,bundle):
        self.config=c
        key='full_matched' if c['method']=='LoRA' else 'uni576' if c['method']=='Uni-LoRA' else 'search_'+str(c['bank'])+'_'+str(c['sparse'])
        if c['method']=='ProLoSA':core.METHODS[key]=(c['bank'],c['sparse'])
        super().__init__(key,bundle)
    def before(self,step):
        c=self.config
        for h in self.holders:h.capture_gradient_stats=bool(self.k and c['warmup']<step<=c['warmup']+c['score_steps'])

def train(model,a,x,y,clean,xe,truth,checkpoints):
    c=a.config
    if a.k:
        opt=torch.optim.AdamW([{'params':[a.bank],'lr':c['lr']},{'params':[a.residual],'lr':c['lr']*c['residual_lr_multiplier']}],weight_decay=0.)
    else:opt=torch.optim.AdamW(a.parameters(),lr=c['lr'],weight_decay=0.)
    g=torch.Generator(device='cuda').manual_seed(81000);preds={};factors={};rows=[];torch.cuda.synchronize();started=time.perf_counter()
    for step in range(1,max(checkpoints)+1):
        a.before(step);ix=torch.randint(len(x),(128,),generator=g,device='cuda')
        loss=F.mse_loss(model(x[ix],a()),y[ix]);opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(a.parameters(),1.);opt.step()
        if a.k and step==c['warmup']+c['score_steps']:a.activate()
        if step in checkpoints:
            torch.cuda.synchronize();seconds=time.perf_counter()-started
            with torch.no_grad():
                pred=base.predict(model,xe,a()).cpu().numpy();tr=base.predict(model,x,a());factors[f'step{step}']=a.vector().cpu().numpy()
            assert np.isfinite(pred).all();preds[f'step{step}']=pred
            rows.append(dict(step=step,risk=float(np.square(pred.astype(float)-truth).mean()),noisy_train_mse=float(F.mse_loss(tr,y)),clean_train_mse=float(F.mse_loss(tr,clean)),seconds=seconds))
    return preds,factors,rows

def source_audit(bundle,model):
    c=dict(method='ProLoSA',bank=288,sparse=288,lr=.0002,residual_lr_multiplier=1.,warmup=128,score_steps=128)
    a=Adapter(c,bundle);b=core.Adapter('pro288_288',bundle)
    torch.testing.assert_close(a.vector(),b.vector(),rtol=0,atol=0)
    g=torch.Generator(device='cuda').manual_seed(990099);x=torch.randn(256,32,64,generator=g,device='cuda')
    clean=base.predict(model,x,{k:v.cuda() for k,v in bundle['teacher'].items()});y=clean+.05*torch.randn(256,16,generator=g,device='cuda')
    truth=clean[:32].cpu().numpy().astype(float)
    pa,fa,_=train(model,a,x,y,clean,x[:32],truth,[1000]);pb,fb,_=core.train(model,b,x,y,clean,x[:32],truth,.0002,[1000])
    np.testing.assert_allclose(pa['step1000'],pb['step1000'],rtol=1e-5,atol=1e-6)
    np.testing.assert_allclose(fa['step1000'],fb['step1000'],rtol=1e-5,atol=1e-6)
    assert torch.equal(a.mask,b.mask)
    return dict(standard_configuration_matches_prior_training=True,prediction_max_error=float(np.max(np.abs(pa['step1000']-pb['step1000']))),factor_max_error=float(np.max(np.abs(fa['step1000']-fb['step1000']))),support_exact=True)

def require():
    p=json.loads((HERE/'prepared.json').read_text())
    for key,path in [('source_sha256',Path(__file__)),('plan_sha256',HERE/'PLAN.md'),('prior_prepared_sha256',PRIOR/'prepared.json'),('bundle_sha256',HERE/'prepared.pt'),('teacher_source_sha256',HERE/'teacher.py'),('teacher_diagnostics_sha256',HERE/'teacher_diagnostics.json')]:assert p[key]==sha(path),key
    core.require()
    return p,torch.load(HERE/'prepared.pt',map_location='cpu',weights_only=False)

def prepare():
    from teacher import construct
    assert not (HERE/'prepared.json').exists();p,bundle=core.require();model=core.model_load()
    bundle.pop('teacher_factors',None)
    tensors,info=construct(core,model)
    bundle.update(tensors)
    for d in [461,512,542]:bundle['banks'][d]=core.init_bank(d)
    audit=source_audit(bundle,model)
    audit.update(core.audit(bundle,model))
    xdev=torch.randn(1024,32,64,generator=torch.Generator(device='cuda').manual_seed(1000040),device='cuda')
    bundle['dev_x']=xdev.cpu();bundle['dev_truth']=base.predict(model,xdev,{k:v.cuda() for k,v in bundle['teacher'].items()}).cpu()
    torch.save(bundle,HERE/'prepared.pt');cc=configs()
    grid=[dict(config_id=cid,method=c['method'],noise_multiple=s,repeat=r) for cid,c in cc.items() for s in [0.,1.,4.] for r in range(3)]
    assert len(cc)==114 and len(grid)==1026
    atomic(HERE/'teacher_diagnostics.json',info)
    atomic(HERE/'prepared.json',dict(source_sha256=sha(__file__),teacher_source_sha256=sha(HERE/'teacher.py'),teacher_diagnostics_sha256=sha(HERE/'teacher_diagnostics.json'),plan_sha256=sha(HERE/'PLAN.md'),prior_prepared_sha256=sha(PRIOR/'prepared.json'),bundle_sha256=sha(HERE/'prepared.pt'),target_id=core.delta_digest(bundle['teacher']),noise_unit=info['noise_unit'],teacher=info,configs=cc,dev_grid=grid,audit=audit))
    print('Prepared weight spike-slab teacher',info,flush=True)

def run(mode,shard,shards):
    p,bundle=require();model=core.model_load();teacher={k:v.cuda() for k,v in bundle['teacher'].items()}
    if mode=='main':
        sel=json.loads((HERE/'selection.json').read_text());assert sel['prepared_sha256']==sha(HERE/'prepared.json') and sel['test_sha256']==sha(HERE/'test.pt')
        tests=torch.load(HERE/'test.pt',map_location='cpu',weights_only=False);xe=tests['x'].cuda();truth=tests['truth'].numpy().astype(float);grid=sel['main_grid']
    else:xe=bundle['dev_x'].cuda();truth=bundle['dev_truth'].numpy().astype(float);grid=p['dev_grid']
    for i in range(shard,len(grid),shards):
        spec=grid[i];c=p['configs'][spec['config_id']];out=HERE/mode/f'{i:04d}';out.mkdir(parents=True,exist_ok=True)
        if (out/'result.json').exists():
            r=json.loads((out/'result.json').read_text());assert r['source_sha256']==sha(__file__) and r['spec']==spec
            for f in ['predictions','factors','support']:assert r[f+'_sha256']==sha(out/(f+'.npz'))
            continue
        g=torch.Generator(device='cuda').manual_seed((1002000 if mode=='main' else 1001000)+spec['repeat'])
        x=torch.randn(256,32,64,generator=g,device='cuda');noise=torch.randn(256,16,generator=g,device='cuda');clean=base.predict(model,x,teacher)
        sigma=spec['noise_multiple']*p['noise_unit'];y=clean+sigma*noise;a=Adapter(c,bundle)
        checkpoints=[1000,3000] if mode=='main' and spec['long_control'] else [1000]
        preds,factors,rows=train(model,a,x,y,clean,xe,truth,checkpoints)
        np.savez_compressed(out/'predictions.npz',**preds);np.savez_compressed(out/'factors.npz',**factors)
        support=dict(mask=a.mask.cpu().numpy(),scores=a.scores.cpu().numpy())
        if hasattr(a,'bank'):support.update(bank=a.bank.detach().cpu().numpy(),residual=a.residual.detach().cpu().numpy())
        np.savez_compressed(out/'support.npz',**support)
        r=dict(spec=spec,config=c,mode=mode,lr=c['lr'],sigma=sigma,rows=rows,source_sha256=sha(__file__),prepared_sha256=sha(HERE/'prepared.json'),target_id=p['target_id'],test_id=base.digest(xe.cpu().numpy()),train_x_id=base.digest(x.cpu().numpy()),noise_id=base.digest(noise.cpu().numpy()),train_y_id=base.digest(y.cpu().numpy()),effective_parameters=4608 if c['method']=='LoRA' else c['bank']+c['sparse'],gpu=torch.cuda.get_device_name(0),torch_version=torch.__version__,job=os.environ['SLURM_JOB_ID'])
        for f in ['predictions','factors','support']:r[f+'_sha256']=sha(out/(f+'.npz'))
        if mode=='main':r['selection_sha256']=sha(HERE/'selection.json')
        atomic(out/'result.json',r);print(mode,i,spec,rows,flush=True)

def lock():
    p,bundle=require();assert not (HERE/'selection.json').exists();records=[]
    for i,spec in enumerate(p['dev_grid']):
        out=HERE/'dev'/f'{i:04d}';r=json.loads((out/'result.json').read_text());assert r['spec']==spec and r['config']==p['configs'][spec['config_id']] and r['source_sha256']==sha(__file__) and r['predictions_sha256']==sha(out/'predictions.npz');records.append(r)
    scores=[]
    for cid,c in p['configs'].items():
        rr=[r for r in records if r['spec']['config_id']==cid];assert len(rr)==9
        scores.append(dict(config_id=cid,**c,mean_dev_mse=float(np.mean([r['rows'][0]['risk'] for r in rr])),dev_fits=9))
    selected={m:min([r for r in scores if r['method']==m],key=lambda r:(r['mean_dev_mse'],r['config_id']))['config_id'] for m in METHODS}
    atomic(HERE/'dev_selection_before_test.json',dict(selected=selected,scores=scores,prepared_sha256=sha(HERE/'prepared.json')))
    model=core.model_load();x=torch.randn(1024,32,64,generator=torch.Generator(device='cuda').manual_seed(1000050),device='cuda');truth=base.predict(model,x,{k:v.cuda() for k,v in bundle['teacher'].items()}).cpu()
    torch.save(dict(x=x.cpu(),truth=truth),HERE/'test.pt');np.save(HERE/'truth.npy',truth.numpy())
    grid=[dict(config_id=selected[m],method=m,noise_multiple=s,repeat=r,long_control=s in [1.,4.] and r<5) for s in MULT for m in METHODS for r in range(20)]
    atomic(HERE/'selection.json',dict(selected=selected,selected_configs={m:p['configs'][cid] for m,cid in selected.items()},scores=scores,main_grid=grid,prepared_sha256=sha(HERE/'prepared.json'),dev_selection_sha256=sha(HERE/'dev_selection_before_test.json'),test_sha256=sha(HERE/'test.pt'),truth_sha256=sha(HERE/'truth.npy'),test_id=base.digest(x.cpu().numpy()),dev_results_sha256={f'{i:04d}':sha(HERE/'dev'/f'{i:04d}'/'result.json') for i in range(len(records))}))
    print('LOCKED', {m:(cid,p['configs'][cid]) for m,cid in selected.items()},flush=True)

def control():
    p,bundle=require();model=core.model_load();teacher={k:v.cuda() for k,v in bundle['teacher'].items()}
    sel=json.loads((HERE/'selection.json').read_text());assert sel['prepared_sha256']==sha(HERE/'prepared.json') and sel['test_sha256']==sha(HERE/'test.pt')
    test=torch.load(HERE/'test.pt',map_location='cpu',weights_only=False);xe=test['x'].cuda();truth=test['truth'].numpy().astype(float)
    for rep in range(3):
        g=torch.Generator(device='cuda').manual_seed(1003000+rep);x=torch.randn(4096,32,64,generator=g,device='cuda');clean=base.predict(model,x,teacher)
        for method in METHODS:
            out=HERE/'control'/f'{method}_{rep}';out.mkdir(parents=True,exist_ok=True)
            c=p['configs'][sel['selected'][method]];a=Adapter(c,bundle)
            preds,factors,rows=train(model,a,x,clean,clean,xe,truth,[1000,4000])
            np.savez_compressed(out/'predictions.npz',**preds);np.savez_compressed(out/'factors.npz',**factors)
            atomic(out/'result.json',dict(method=method,repeat=rep,n=4096,sigma=0,config=c,rows=rows,source_sha256=sha(__file__),prepared_sha256=sha(HERE/'prepared.json'),selection_sha256=sha(HERE/'selection.json'),target_id=p['target_id'],train_x_id=base.digest(x.cpu().numpy()),test_id=sel['test_id'],predictions_sha256=sha(out/'predictions.npz'),factors_sha256=sha(out/'factors.npz')))
            print('control',method,rep,rows,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','dev','lock','main','control']);ap.add_argument('--shards',type=int,default=8);a=ap.parse_args();core.old.setup()
    if a.mode=='prepare':prepare()
    elif a.mode=='lock':lock()
    elif a.mode=='control':control()
    else:run(a.mode,int(os.environ.get('SLURM_ARRAY_TASK_ID','0')),a.shards)
