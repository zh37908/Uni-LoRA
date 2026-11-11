"""Frozen one-block Transformer, with rank-factor updates on all six block maps."""
import sys
from pathlib import Path
import torch
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'predictive_bias_variance_20260924'))
import model_frozen as base
TARGETS=[('q',64,64),('k',64,64),('v',64,64),('o',64,64),('ff1',64,256),('ff2',256,64)]
RANK=4
SLICES=[];offset=0
for name,inputs,outputs in TARGETS:
    for factor,shape in [('A',(RANK,inputs)),('B',(outputs,RANK))]:
        size=shape[0]*shape[1];SLICES.append((name,factor,shape,offset,offset+size));offset+=size
D=offset
assert D==4608

def delta_from_factors(matrices):return {name:matrices[2*i+1]@matrices[2*i] for i,(name,_,_) in enumerate(TARGETS)}
def delta_digest(delta):return base.digest(torch.cat([delta[n].detach().cpu().flatten() for n,_,_ in TARGETS]).double().numpy())

class Block(base.Block):
    def forward(self,x,delta=None):
        def linear(name,z):
            layer=getattr(self,name)
            return F.linear(z,layer.weight if delta is None else layer.weight+delta[name],layer.bias)
        q,k,v=[linear(n,x) for n in ['q','k','v']];batch,t,_=x.shape
        q,k,v=[z.reshape(batch,t,4,16).transpose(1,2) for z in [q,k,v]]
        attention=(q@k.transpose(-1,-2)/4).softmax(-1)@v
        y=self.ln1(x+linear('o',attention.transpose(1,2).reshape(batch,t,64)))
        y=self.ln2(y+linear('ff2',F.gelu(linear('ff1',y))))
        return self.readout(y.mean(1))
