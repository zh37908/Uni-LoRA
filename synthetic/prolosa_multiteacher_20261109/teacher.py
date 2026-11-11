"""Projection-independent spike/slab in physical weight coordinates."""
import math
import numpy as np
import torch

def construct(core, model, teacher_seed):
    sizes=[inputs*outputs for _,inputs,outputs in core.TARGETS]
    dimension=sum(sizes)
    assert dimension==49152
    rng=np.random.default_rng(teacher_seed)
    slab=rng.random(dimension)<.01
    gaussian=rng.standard_normal(dimension)
    student=rng.standard_t(3,dimension)
    raw=np.where(slab,.5*math.sqrt(1/3)*student,.005*gaussian)
    weights={}; masks={};start=0
    for (name,inputs,outputs),size in zip(core.TARGETS,sizes):
        weights[name]=torch.tensor(raw[start:start+size].reshape(outputs,inputs),dtype=torch.float32,device='cuda')
        masks[name]=torch.tensor(slab[start:start+size].reshape(outputs,inputs))
        start+=size
    xcal=torch.randn(1024,32,64,generator=torch.Generator(device='cuda').manual_seed(1000030),device='cuda')
    baseline=core.base.predict(model,xcal,None)
    base_variance=float(baseline.var(unbiased=False)); target=.1*base_variance
    def delta(scale):return {k:scale*v for k,v in weights.items()}
    def displacement(scale):return float((core.base.predict(model,xcal,delta(scale))-baseline).square().mean())
    lo,hi=0.,1.
    while displacement(hi)<target:
        hi*=2
        assert hi<1e5
    for _ in range(32):
        mid=(lo+hi)/2
        if displacement(mid)<target:lo=mid
        else:hi=mid
    scale=(lo+hi)/2;teacher=delta(scale);actual=displacement(scale)
    assert abs(actual/target-1)<2e-5
    truthcal=core.base.predict(model,xcal,teacher);effects={};r4teacher={}
    for name,_,_ in core.TARGETS:
        w=teacher[name];a,s,v=torch.linalg.svd(w.double(),full_matrices=False)
        rank4=(a[:,:4]*s[:4])@v[:4];r4teacher[name]=rank4.float()
        ablated=dict(teacher);ablated[name]=torch.zeros_like(w)
        impact=float((core.base.predict(model,xcal,ablated)-truthcal).square().mean())
        assert impact>0
        effects[name]=dict(elements=w.numel(),slab_count=int(masks[name].sum()),update_frobenius=float(w.norm()),base_weight_frobenius=float(getattr(model,name).weight.norm()),numerical_rank=int(torch.linalg.matrix_rank(w.double())),rank4_weight_residual_fraction=float(s[4:].square().sum()/s.square().sum()),leave_one_out_prediction_mse=impact)
    energy=raw**2;order=np.sort(energy)[::-1];k=math.ceil(.01*dimension)
    info=dict(coordinates='physical weight deltas, not LoRA factors',dimension=dimension,seed=teacher_seed,pi=.01,tau0=.005,tau1=.5,nu=3,unscaled_second_moment=.00252475,scale=scale,scaled_spike_sd=.005*scale,scaled_slab_sd=.5*scale,slab_count=int(slab.sum()),slab_fraction=float(slab.mean()),slab_energy_fraction=float(energy[slab].sum()/energy.sum()),top_1pct_count=k,top_1pct_energy_fraction=float(order[:k].sum()/energy.sum()),absolute_delta_quantiles={str(q):float(np.quantile(np.abs(scale*raw),q)) for q in [.5,.9,.99,.999,1.]},modules=effects,base_variance=base_variance,displacement=actual,noise_unit=math.sqrt(actual),rank4_svd_prediction_mse=float((core.base.predict(model,xcal,r4teacher)-truthcal).square().mean()),rank4_svd_note='Weight Frobenius-optimal SVD is not a prediction-optimal lower bound.')
    tensors=dict(teacher={k:v.cpu() for k,v in teacher.items()},teacher_slab_masks=masks,teacher_raw=torch.from_numpy(raw),calibration_inputs=xcal.cpu())
    return tensors,info
