"""Revision §5: fixed-teacher Transformer controls (new, explicit protocol).
One immutable prepared teacher per profile, equal functional displacement,
common P/test inputs and training samples, matched B=128 compressed/hybrid
budgets. Support-quality hybrids use a pilot followed by a fresh fit; same-data
and independent-data variants share this reset, so independence is not
confounded with transferring pilot parameters. These are diagnostic hybrids,
not mislabeled as the original end-to-end ProLoSA training protocol.
"""
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]/'results_revision5_fixed_teacher'
METHODS=['lora','compressed','random','gradient','snip','energy','snip_independent','snip_same_half']
D=8192;WIDTH=64;d=64;K=64;B=128
STEPS=1000;PILOT=256;SCORE=128

def digest(a):return hashlib.sha256(np.asarray(a,dtype='<f8').tobytes()).hexdigest()

def seed_all(s):
    torch.manual_seed(s);np.random.seed(s);torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.q=nn.Linear(64,64,bias=False);self.k=nn.Linear(64,64,bias=False)
        self.v=nn.Linear(64,64,bias=False);self.o=nn.Linear(64,64,bias=False)
        self.ln1=nn.LayerNorm(64);self.ln2=nn.LayerNorm(64)
        self.ff1=nn.Linear(64,256);self.ff2=nn.Linear(256,64);self.readout=nn.Linear(64,16,bias=False)
    def forward(self,x,delta=None):
        q=self.q(x) if delta is None else F.linear(x,self.q.weight+delta[:4096].view(64,64))
        v=self.v(x) if delta is None else F.linear(x,self.v.weight+delta[4096:].view(64,64))
        k=self.k(x);n,t,_=x.shape
        q,k,v=[z.reshape(n,t,4,16).transpose(1,2) for z in [q,k,v]]
        att=(q@k.transpose(-1,-2)/4).softmax(-1)@v
        y=self.ln1(x+self.o(att.transpose(1,2).reshape(n,t,64)))
        y=self.ln2(y+self.ff2(F.gelu(self.ff1(y))))
        return self.readout(y.mean(1))

class Adapter(nn.Module):
    def __init__(self,assignment,kind='compressed',support=None,latent=d):
        super().__init__();self.kind=kind
        self.register_buffer('assignment',assignment)
        if kind=='lora':
            self.A=nn.Parameter(torch.randn(2,4,64,device=assignment.device)*.02)
            self.B=nn.Parameter(torch.zeros(2,64,4,device=assignment.device))
        else:
            self.z=nn.Parameter(torch.zeros(latent,device=assignment.device))
            self.register_buffer('scale',torch.bincount(assignment,minlength=latent).float().sqrt())
            self.register_buffer('support',torch.empty(0,dtype=torch.long,device=assignment.device) if support is None else support)
            self.alpha=nn.Parameter(torch.zeros(len(self.support),device=assignment.device))
    def forward(self):
        if self.kind=='lora':return (self.B@self.A).reshape(-1)
        val=(self.z/self.scale)[self.assignment]
        return val.scatter_add(0,self.support,self.alpha)

def predict(model,x,delta):
    with torch.no_grad():return torch.cat([model(b,delta) for b in x.split(128)])

def prepare(device):
    ROOT.mkdir(parents=True,exist_ok=True)
    path=ROOT/'teachers.pt'
    if path.exists():print('Prepared teachers already exist:',path);return
    seed_all(13);model=Block().to(device)
    g=torch.Generator(device=device).manual_seed(10013)
    x=torch.randn(4096,32,64,generator=g,device=device)
    mat=torch.randn(64,16,generator=g,device=device)/8
    y=torch.sin(x.mean(1)@mat)
    opt=torch.optim.AdamW(model.parameters(),lr=.002)
    for step in range(300):
        ix=torch.randint(len(x),(128,),generator=g,device=device);loss=F.mse_loss(model(x[ix]),y[ix]);opt.zero_grad();loss.backward();opt.step()
    model.eval().requires_grad_(False)
    xcal=torch.randn(1024,32,64,generator=g,device=device)
    xtest=torch.randn(1024,32,64,generator=g,device=device)
    base=predict(model,xcal,None);var=float(base.var(unbiased=False))
    rng=np.random.default_rng(913);perm=rng.permutation(D);assignment=perm%d;matched=perm%B
    payload=dict(state={k:v.cpu() for k,v in model.state_dict().items()},x_test=xtest.cpu(),
        assignment=torch.tensor(assignment),matched_assignment=torch.tensor(matched),profiles={},
        output_variance=var,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    for profile in ['concentrated','diffuse']:
        direction=rng.standard_normal(D)
        if profile=='concentrated':
            direction*=.005;indices=rng.choice(D,64,replace=False);direction[indices]=rng.standard_t(3,64)
        direction=torch.tensor(direction/np.linalg.norm(direction),dtype=torch.float32,device=device)
        target=.1*var
        def disp(scale):return float((predict(model,xcal,direction*scale)-base).square().mean())
        lo,hi=0.,1.
        while disp(hi)<target:
            hi*=2
            if hi>1e4:raise RuntimeError('Cannot calibrate teacher displacement')
        for _ in range(28):
            mid=(lo+hi)/2
            if disp(mid)<target:lo=mid
            else:hi=mid
        delta=direction*((lo+hi)/2);actual=disp((lo+hi)/2)
        assert abs(actual/target-1)<1e-4
        theta=delta.cpu().double().numpy();means=np.bincount(assignment,weights=theta)/np.bincount(assignment)
        q=theta-means[assignment];capture=np.sort(q*q)[-K:].sum()/(q@q)
        payload['profiles'][profile]=dict(theta=delta.cpu(),y_test=predict(model,xtest,delta).cpu(),target_id=digest(theta),
            residual_topk_fraction=float(capture),displacement=actual,displacement_target=target)
    assert payload['profiles']['concentrated']['residual_topk_fraction']>payload['profiles']['diffuse']['residual_topk_fraction']
    torch.save(payload,path)
    summary={k:{a:b for a,b in v.items() if a not in ['theta','y_test']} for k,v in payload['profiles'].items()}
    (ROOT/'teacher_manifest.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2),flush=True)

def fit(model,adapter,x,y,steps,seed,score=False):
    opt=torch.optim.AdamW(adapter.parameters(),lr=.005,weight_decay=0)
    g=torch.Generator(device=x.device).manual_seed(seed)
    grad=torch.zeros(D,device=x.device);snip=grad.clone()
    for t in range(steps):
        ix=torch.randint(len(x),(min(128,len(x)),),generator=g,device=x.device)
        delta=adapter()
        if score and t>=steps-min(SCORE,steps):delta.retain_grad()
        loss=F.mse_loss(model(x[ix],delta),y[ix]);opt.zero_grad();loss.backward()
        if score and t>=steps-min(SCORE,steps):
            grad+=delta.grad.detach().abs();snip+=(delta.detach()*delta.grad.detach()).abs()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(),1);opt.step()
    return grad,snip

def run(index,smoke,device):
    grid=[(p,n,sig,s) for p in ['concentrated','diffuse'] for n in [64,256,1024,4096] for sig in [.25,.5,1.] for s in range(20)]
    profile,n,sigma,seed=grid[index];out=ROOT/('smoke_v2' if smoke else 'trials_v2')/f'{index:04d}';out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():return
    payload=torch.load(ROOT/'teachers.pt',map_location='cpu',weights_only=False)
    frozen=payload['profiles'][profile];theta=frozen['theta'].to(device);target_id=digest(theta.cpu().double().numpy())
    assert target_id==frozen['target_id']
    model=Block().to(device);model.load_state_dict(payload['state']);model.eval().requires_grad_(False)
    assignment=payload['assignment'].to(device);matched=payload['matched_assignment'].to(device)
    # Same seed/n shares X and noise across profiles and all support methods.
    g=torch.Generator(device=device).manual_seed(70000+seed*100+n)
    x=torch.randn(n,32,64,generator=g,device=device)
    y=predict(model,x,theta)+sigma*torch.randn(n,16,generator=g,device=device)
    xtest=payload['x_test'].to(device);truth=frozen['y_test'].numpy()
    half=n//2;pilot_steps=4 if smoke else PILOT;total_steps=12 if smoke else STEPS
    seed_all(5000+seed);pilot=Adapter(assignment).to(device);torch.cuda.synchronize();t0=time.perf_counter()
    grad,snip=fit(model,pilot,x,y,pilot_steps,80000+seed,score=True);torch.cuda.synchronize();pilot_time=time.perf_counter()-t0
    # Independent selection: ONLY A calibrates support; B trains a fresh model.
    # No pilot coefficients/optimizer state are transferred from A to B.
    seed_all(5000+seed);ind_pilot=Adapter(assignment).to(device);torch.cuda.synchronize();t0=time.perf_counter()
    _,ind_snip=fit(model,ind_pilot,x[:half],y[:half],pilot_steps,80000+seed,score=True);torch.cuda.synchronize();ind_time=time.perf_counter()-t0
    # Matched fitting-size comparator: choose support on B and fit on B.
    seed_all(5000+seed);half_pilot=Adapter(assignment).to(device);torch.cuda.synchronize();t0=time.perf_counter()
    _,half_snip=fit(model,half_pilot,x[half:],y[half:],pilot_steps,80000+seed,score=True);torch.cuda.synchronize();half_time=time.perf_counter()-t0
    q=theta-torch.bincount(assignment,weights=theta,minlength=d)[assignment]/torch.bincount(assignment)[assignment]
    gen=torch.Generator(device=device).manual_seed(90000+seed)
    masks=dict(random=torch.randperm(D,generator=gen,device=device)[:K],gradient=grad.topk(K).indices,
        snip=snip.topk(K).indices,energy=q.square().topk(K).indices,snip_independent=ind_snip.topk(K).indices,snip_same_half=half_snip.topk(K).indices)
    rows=[];preds={}
    for method in METHODS:
        seed_all(6000+seed)
        if method=='lora':adapter=Adapter(assignment,kind='lora').to(device)
        elif method=='compressed':adapter=Adapter(matched,latent=B).to(device)
        else:adapter=Adapter(assignment,support=masks[method]).to(device)
        independent=method=='snip_independent';half_fit=method in ['snip_independent','snip_same_half'];xx,yy=(x[half:],y[half:]) if half_fit else (x,y)
        steps=total_steps if method in ['lora','compressed'] else total_steps-pilot_steps
        torch.cuda.synchronize();t0=time.perf_counter();fit(model,adapter,xx,yy,steps,81000+seed)
        torch.cuda.synchronize();elapsed=time.perf_counter()-t0+(0 if method in ['lora','compressed'] else ind_time if independent else half_time if half_fit else pilot_time)
        pred=predict(model,xtest,adapter()).cpu().numpy();preds[method]=pred
        mask=masks.get(method);capture=None if mask is None else float(q[mask].square().sum()/q.square().sum())
        rows.append(dict(method=method,mse=float(np.mean((pred.astype(float)-truth)**2)),residual_capture=capture,
            params=sum(v.numel() for v in adapter.parameters()),selection_examples=half if half_fit else n if mask is not None else 0,
            fitting_examples=len(xx),total_unique_examples=half if method=='snip_same_half' else n,available_examples=n,charged_gradient_steps=total_steps,charged_seconds=elapsed,
            support=[] if mask is None else mask.cpu().tolist()))
        assert method=='lora' or rows[-1]['params']==B
        print(profile,n,sigma,seed,method,rows[-1]['mse'],flush=True)
    assert digest(theta.cpu().double().numpy())==target_id
    np.savez_compressed(out/'predictions.npz',**preds)
    (out/'result.json').write_text(json.dumps(dict(profile=profile,n=n,noise_std=sigma,seed=seed,target_id=target_id,
        test_id=digest(payload['x_test'].numpy()),rows=rows,smoke=smoke,
        protocol='pilot_reset_matched_half_v2',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--index',type=int,default=0);ap.add_argument('--smoke',action='store_true');ap.add_argument('--count',type=int,default=1);ap.add_argument('--shards',type=int,default=0);args=ap.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('Submit this experiment through Slurm to a GPU node')
    torch.set_num_threads(4)
    if args.prepare:prepare('cuda')
    else:
        for index in (range(args.index,480,args.shards) if args.shards else range(args.index*args.count,(args.index+1)*args.count)):run(index,args.smoke,'cuda')
