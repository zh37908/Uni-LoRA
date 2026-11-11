import sys,json,os,types,time,argparse,math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'lora_factor_mechanism_20260924'))
import run_factors as old
from model_all_linear import Block,TARGETS,SLICES,D,delta_from_factors,delta_digest
base,sha,atomic=old.base,old.sha,old.atomic
METHODS={'full_matched':(0,0),'full_zero':(0,0),'uni288':(288,0),'uni576':(576,0),'uni864':(864,0),'pro288_288':(288,288),'pro576_288':(576,288)}
LRS=[.0002,.001,.005,.02];MULT=[0.,.5,1.,2.,4.]
SRC=HERE/'prolosa_layer_source.py'
GET=old.source_method(SRC,'Linear','_get_lora_matrices')
HOOK=old.source_method(SRC,'UniLoRARoSALayer','_register_gradient_capture_hook')
ACTIVE=old.source_method(SRC,'UniLoRARoSALayer','set_sparse_active')
HAS=old.source_method(SRC,'UniLoRARoSALayer','has_sparse_mask')
GENERATE=old.source_method(HERE/'unilora_model_source.py','UniLoRAModel','generate_index')
UNIGET=old.source_method(HERE/'unilora_layer_source.py','Linear','_get_lora_matrices')

def model_load():
    payload=torch.load(old.TEACHER,map_location='cpu',weights_only=False)
    model=Block().cuda();model.load_state_dict(payload['state']);return model.eval().requires_grad_(False)

def init_bank(d):
    assignment=GENERATE(None,D,d,42)
    scales=1/torch.bincount(assignment,minlength=d).float().sqrt()[assignment]
    bank=torch.empty(d,device='cuda').uniform_(-.02,.02,generator=torch.Generator(device='cuda').manual_seed(980010))
    return dict(bank=bank.cpu(),assignment=assignment,scales=scales)

class Adapter(nn.Module):
    def __init__(self,method,bundle):
        super().__init__();self.method=method;d,self.k=METHODS[method];self.holders=[]
        self.register_buffer('mask',torch.zeros(D,device='cuda',dtype=torch.bool));self.register_buffer('scores',torch.zeros(D,device='cuda'))
        if not d:
            self.params=nn.ParameterList([nn.Parameter(t.cuda().clone()) for t in bundle['matched' if method=='full_matched' else 'standard']]);return
        init=bundle['banks'][d];self.bank=nn.Parameter(init['bank'].cuda().clone())
        self.register_buffer('assignment',init['assignment'].cuda());self.register_buffer('scales',init['scales'].cuda())
        self.residual=nn.Parameter(torch.zeros(D,device='cuda'),requires_grad=False)
        for i,(name,_,_) in enumerate(TARGETS):
            h=types.SimpleNamespace(unilora_rosa_theta_d={'default':self.bank},unilora_theta_d={'default':self.bank},
                unilora_rosa_sparse_theta_D={'default':self.residual},unilora_rosa_sparse_mask={'default':self.mask},
                unilora_rosa_grad_accum={'default':self.scores},sparse_active={'default':False},
                sparse_flat_indices_A={},sparse_flat_indices_B={},sparse_theta_offsets_A_active={},sparse_theta_offsets_B_active={},
                capture_gradient_stats=False,training=True,gradient_capture_mode='snip',rosa_score_aggregation={'default':'sum'},_capture_update_count={})
            for _,factor,shape,start,end in SLICES[2*i:2*i+2]:
                setattr(h,'unilora_indices_'+factor,{'default':self.assignment[start:end].reshape(shape)})
                setattr(h,'unilora_scales_'+factor,{'default':self.scales[start:end].reshape(shape)})
                setattr(h,'unilora_theta_D_offsets_'+factor,{'default':torch.arange(start,end,device='cuda').reshape(shape)})
            for key,fn in [('has_sparse_mask',HAS),('set_sparse_active',ACTIVE),('_register_gradient_capture_hook',HOOK)]:setattr(h,key,types.MethodType(fn,h))
            self.holders.append(h)
    def matrices(self):
        if hasattr(self,'params'):return list(self.params)
        return [t for h in self.holders for t in (GET if self.k else UNIGET)(h,'default')]
    def vector(self):return torch.cat([t.flatten() for t in self.matrices()])
    def forward(self):return delta_from_factors(self.matrices())
    def before(self,step):
        for h in self.holders:h.capture_gradient_stats=bool(self.k and 129<=step<=256)
    def activate(self):
        assert self.scores.max()>0
        self.mask[torch.topk(self.scores,k=self.k,sorted=False).indices]=True
        self.residual.requires_grad_(True)
        for h in self.holders:h.set_sparse_active('default',True);h.capture_gradient_stats=False

def audit(bundle,model):
    g=torch.Generator(device='cuda').manual_seed(980099);x=torch.randn(8,32,64,generator=g,device='cuda')
    previous=old.model_load();torch.testing.assert_close(previous(x),model(x),rtol=0,atol=0)
    zero={name:torch.zeros(outputs,inputs,device='cuda') for name,inputs,outputs in TARGETS}
    torch.testing.assert_close(model(x),model(x,zero),rtol=0,atol=0)
    u=Adapter('uni576',bundle);p=Adapter('pro576_288',bundle);full=Adapter('full_matched',bundle)
    for a in [p,full]:
        torch.testing.assert_close(u.vector(),a.vector(),rtol=0,atol=0);torch.testing.assert_close(model(x,u()),model(x,a()),rtol=0,atol=0)
    probe=torch.randn(8,16,generator=g,device='cuda');(model(x,full())*probe).sum().backward()
    gradient_norms={name:float(full.params[2*i].grad.norm()+full.params[2*i+1].grad.norm()) for i,(name,_,_) in enumerate(TARGETS)}
    assert min(gradient_norms.values())>0
    assert not any(t.requires_grad for t in model.parameters())
    p.before(129)
    for _ in range(2):p.vector().square().sum().backward()
    initial=(p.bank[p.assignment]*p.scales).detach();torch.testing.assert_close(p.scores,4*initial.square(),rtol=1e-6,atol=1e-10)
    p.activate()
    with torch.no_grad():p.residual[p.mask]=.003
    actual=p.vector();expected=p.bank[p.assignment]*p.scales+p.residual*p.mask
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    ga=torch.autograd.grad(actual.square().sum(),[p.bank,p.residual]);gb=torch.autograd.grad(expected.square().sum(),[p.bank,p.residual])
    for a,b in zip(ga,gb):torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-8)
    assert not ga[1][~p.mask].any()
    # Independent residual-path calculation checks every rectangular module.
    forward_errors={};gradient_errors={}
    matrices=p.matrices()
    for i,(name,inputs,outputs) in enumerate(TARGETS):
        layer=getattr(model,name);xx=torch.randn(3,5,inputs,generator=g,device='cuda');a,b=matrices[2*i:2*i+2]
        first=F.linear(xx,layer.weight,layer.bias)+F.linear(F.linear(xx,a),b)
        second=F.linear(xx,layer.weight+b@a,layer.bias)
        err=float((first-second).abs().max());assert err<3e-6;forward_errors[name]=err
        probe=torch.randn(first.shape,generator=g,device='cuda')
        g1=torch.autograd.grad((first*probe).sum(),p.bank,retain_graph=True)[0]
        g2=torch.autograd.grad((second*probe).sum(),p.bank,retain_graph=True)[0]
        rel=float((g1-g2).norm()/g1.norm());assert rel<3e-5;gradient_errors[name]=rel
    return dict(base_and_zero_update_exact=True,matched_initial_factors_and_predictions_exact=True,same_bank_uni_pro_initial_exact=True,all_six_factor_gradient_norms=gradient_norms,source_factor_and_gradient_equivalence=True,snip_sum_verified=True,unselected_gradient_zero=True,full_parameters=D,forward_errors=forward_errors,bank_gradient_relative_errors=gradient_errors)

def files():return {'source':Path(__file__),'architecture':HERE/'model_all_linear.py','plan':HERE/'PLAN.md','pro_layer':SRC,'pro_model':HERE/'prolosa_model_source.py','uni_layer':HERE/'unilora_layer_source.py','uni_model':HERE/'unilora_model_source.py','frozen_base_source':Path(base.__file__),'base_checkpoint':old.TEACHER,'shared_runner':Path(old.__file__)}
def require():
    p=json.loads((HERE/'prepared.json').read_text())
    for k,f in files().items():assert p['hashes'][k]==sha(f),k
    assert p['bundle_sha256']==sha(HERE/'prepared.pt')
    return p,torch.load(HERE/'prepared.pt',map_location='cpu',weights_only=False)

def prepare():
    assert not (HERE/'prepared.json').exists();model=model_load();banks={d:init_bank(d) for d in [288,576,864]}
    init=banks[576];flat=init['bank'][init['assignment']]*init['scales'];matched=[flat[start:end].reshape(shape).clone() for _,_,shape,start,end in SLICES]
    standard=[];g=torch.Generator(device='cuda').manual_seed(980011)
    for _,factor,shape,_,_ in SLICES:
        t=torch.empty(shape,device='cuda')
        if factor=='A':nn.init.kaiming_uniform_(t,a=math.sqrt(5),generator=g)
        else:t.zero_()
        standard.append(t.cpu())
    bundle=dict(banks=banks,matched=matched,standard=standard);checks=audit(bundle,model)
    g=torch.Generator(device='cuda').manual_seed(980020)
    raw=[torch.randn(shape,generator=g,device='cuda')/math.sqrt(shape[1]) for _,_,shape,_,_ in SLICES]
    xcal=torch.randn(1024,32,64,generator=torch.Generator(device='cuda').manual_seed(980030),device='cuda')
    baseline=base.predict(model,xcal,None);target=.1*float(baseline.var(unbiased=False))
    def delta(s):return delta_from_factors([t*math.sqrt(s) for t in raw])
    def displacement(s):return float((base.predict(model,xcal,delta(s))-baseline).square().mean())
    lo,hi=0.,1.
    while displacement(hi)<target:hi*=2;assert hi<1e5
    for _ in range(30):
        mid=(lo+hi)/2
        if displacement(mid)<target:lo=mid
        else:hi=mid
    scale=(lo+hi)/2;teacher=delta(scale);actual=displacement(scale);assert abs(actual/target-1)<2e-5
    truthcal=base.predict(model,xcal,teacher);effects={}
    for name,_,_ in TARGETS:
        ablated=dict(teacher);ablated[name]=torch.zeros_like(teacher[name]);effect=float((base.predict(model,xcal,ablated)-truthcal).square().mean())
        assert effect>0;effects[name]=dict(update_frobenius=float(teacher[name].norm()),leave_one_out_prediction_mse=effect)
    xdev=torch.randn(1024,32,64,generator=torch.Generator(device='cuda').manual_seed(980040),device='cuda')
    bundle.update(teacher={k:v.cpu() for k,v in teacher.items()},teacher_factors=[(t*math.sqrt(scale)).cpu() for t in raw],calibration_inputs=xcal.cpu(),dev_x=xdev.cpu(),dev_truth=base.predict(model,xdev,teacher).cpu())
    torch.save(bundle,HERE/'prepared.pt')
    dev=[dict(method=m,lr=lr,noise_multiple=s,repeat=r) for m in METHODS for lr in LRS for s in [0.,1.,4.] for r in range(3)]
    main=[dict(method=m,noise_multiple=s,repeat=r,long_control=s in [1.,4.] and r<5) for s in MULT for m in METHODS for r in range(20)]
    atomic(HERE/'prepared.json',dict(hashes={k:sha(f) for k,f in files().items()},bundle_sha256=sha(HERE/'prepared.pt'),target_id=delta_digest(teacher),targets=TARGETS,factor_slices=SLICES,full_parameters=D,teacher_scale=scale,noise_unit=math.sqrt(actual),displacement=actual,base_variance=target/.1,teacher_module_effects=effects,dev_grid=dev,main_grid=main,audit=checks))
    print('Prepared all six maps',checks,'teacher module effects',effects,flush=True)

def train(model,a,x,y,clean,xe,truth,lr,checkpoints):
    opt=torch.optim.AdamW(a.parameters(),lr=lr,weight_decay=0.);g=torch.Generator(device='cuda').manual_seed(81000)
    preds={};factors={};rows=[];torch.cuda.synchronize();started=time.perf_counter()
    for step in range(1,max(checkpoints)+1):
        a.before(step);ix=torch.randint(256,(128,),generator=g,device='cuda')
        loss=F.mse_loss(model(x[ix],a()),y[ix]);opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(a.parameters(),1.);opt.step()
        if step==256 and a.k:a.activate()
        if step in checkpoints:
            torch.cuda.synchronize();seconds=time.perf_counter()-started
            with torch.no_grad():
                pred=base.predict(model,xe,a()).cpu().numpy();training=base.predict(model,x,a());factors[f'step{step}']=a.vector().cpu().numpy()
            assert np.isfinite(pred).all();preds[f'step{step}']=pred
            rows.append(dict(step=step,risk=float(np.square(pred.astype(float)-truth).mean()),clean_train_mse=float(F.mse_loss(training,clean)),noisy_train_mse=float(F.mse_loss(training,y)),seconds=seconds))
    return preds,factors,rows

def run(mode,shard,shards):
    p,bundle=require();model=model_load();teacher={k:v.cuda() for k,v in bundle['teacher'].items()}
    if mode=='main':
        selection=json.loads((HERE/'selection.json').read_text());assert selection['prepared_sha256']==sha(HERE/'prepared.json') and selection['test_sha256']==sha(HERE/'test.pt')
        tests=torch.load(HERE/'test.pt',map_location='cpu',weights_only=False);xe=tests['x'].cuda();truth=tests['truth'].numpy().astype(float)
    else:xe=bundle['dev_x'].cuda();truth=bundle['dev_truth'].numpy().astype(float)
    for i in range(shard,len(p[mode+'_grid']),shards):
        spec=p[mode+'_grid'][i];out=HERE/mode/f'{i:04d}';out.mkdir(parents=True,exist_ok=True)
        if (out/'result.json').exists():
            r=json.loads((out/'result.json').read_text());assert r['source_sha256']==sha(__file__) and r['spec']==spec
            for f in ['predictions','factors','support']:assert r[f+'_sha256']==sha(out/(f+'.npz'))
            continue
        lr=selection['learning_rates'][spec['method']] if mode=='main' else spec['lr']
        g=torch.Generator(device='cuda').manual_seed((982000 if mode=='main' else 981000)+spec['repeat'])
        x=torch.randn(256,32,64,generator=g,device='cuda');noise=torch.randn(256,16,generator=g,device='cuda');clean=base.predict(model,x,teacher)
        sigma=spec['noise_multiple']*p['noise_unit'];y=clean+sigma*noise;a=Adapter(spec['method'],bundle)
        checkpoints=[1000,3000] if mode=='main' and spec['long_control'] else [1000]
        preds,factors,rows=train(model,a,x,y,clean,xe,truth,lr,checkpoints)
        np.savez_compressed(out/'predictions.npz',**preds);np.savez_compressed(out/'factors.npz',**factors)
        supports=dict(mask=a.mask.cpu().numpy(),scores=a.scores.cpu().numpy())
        if hasattr(a,'bank'):supports.update(bank=a.bank.detach().cpu().numpy(),residual=a.residual.detach().cpu().numpy())
        np.savez_compressed(out/'support.npz',**supports)
        r=dict(spec=spec,mode=mode,lr=lr,sigma=sigma,rows=rows,source_sha256=sha(__file__),prepared_sha256=sha(HERE/'prepared.json'),target_id=p['target_id'],test_id=base.digest(xe.cpu().numpy()),train_x_id=base.digest(x.cpu().numpy()),noise_id=base.digest(noise.cpu().numpy()),train_y_id=base.digest(y.cpu().numpy()),effective_parameters=sum(METHODS[spec['method']]) or D,allocated_parameter_elements=sum(t.numel() for t in a.parameters()),gpu=torch.cuda.get_device_name(0),torch_version=torch.__version__,job=os.environ['SLURM_JOB_ID'])
        for f in ['predictions','factors','support']:r[f+'_sha256']=sha(out/(f+'.npz'))
        if mode=='main':r['selection_sha256']=sha(HERE/'selection.json')
        atomic(out/'result.json',r);print(mode,i,spec,lr,rows,flush=True)

def lock():
    p,bundle=require();assert not (HERE/'selection.json').exists();records=[]
    for i,spec in enumerate(p['dev_grid']):
        out=HERE/'dev'/f'{i:04d}';r=json.loads((out/'result.json').read_text());assert r['spec']==spec and r['source_sha256']==sha(__file__) and r['predictions_sha256']==sha(out/'predictions.npz');records.append(r)
    scores=[];selected={}
    for m in METHODS:
        for lr in LRS:
            rr=[r for r in records if r['spec']['method']==m and r['lr']==lr];assert len(rr)==9
            scores.append(dict(method=m,lr=lr,mean_dev_mse=float(np.mean([r['rows'][0]['risk'] for r in rr]))))
        selected[m]=min([r for r in scores if r['method']==m],key=lambda r:(r['mean_dev_mse'],r['lr']))['lr']
    atomic(HERE/'dev_selection_before_test.json',dict(learning_rates=selected,scores=scores,prepared_sha256=sha(HERE/'prepared.json')))
    model=model_load();x=torch.randn(1024,32,64,generator=torch.Generator(device='cuda').manual_seed(980050),device='cuda')
    truth=base.predict(model,x,{k:v.cuda() for k,v in bundle['teacher'].items()}).cpu()
    torch.save(dict(x=x.cpu(),truth=truth),HERE/'test.pt');np.save(HERE/'truth.npy',truth.numpy())
    atomic(HERE/'selection.json',dict(learning_rates=selected,scores=scores,prepared_sha256=sha(HERE/'prepared.json'),dev_selection_sha256=sha(HERE/'dev_selection_before_test.json'),test_sha256=sha(HERE/'test.pt'),truth_sha256=sha(HERE/'truth.npy'),test_id=base.digest(x.cpu().numpy()),dev_results_sha256={f'{i:04d}':sha(HERE/'dev'/f'{i:04d}'/'result.json') for i in range(len(records))}))
    print('LOCKED before generating test',selected,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','dev','lock','main']);p.add_argument('--shards',type=int,default=8);a=p.parse_args();old.setup()
    if a.mode=='prepare':prepare()
    elif a.mode=='lock':lock()
    else:run(a.mode,int(os.environ.get('SLURM_ARRAY_TASK_ID','0')),a.shards)
