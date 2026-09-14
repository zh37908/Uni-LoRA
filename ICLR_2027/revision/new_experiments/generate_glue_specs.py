from pathlib import Path
import itertools,json
P=Path(__file__).resolve().parent;S=P/'specs';S.mkdir(exist_ok=True)
# 12 balanced configurations per method/task: 3 learning rates x 4 regularizer /
# stopping tuples; head learning rate is tied identically across method families.
reg=[(0.,0.,'best'),(.01,.05,'best'),(.1,.1,'best'),(.01,.05,'last')]
rows=[]
for task,method,lr_i,reg_i,seed in itertools.product(['cola','mrpc'],['lora','unilora','prolosa','rosa'],range(3),range(4),[101,102]):
    lr=([1e-4,4e-4,1e-3] if method in ['lora','rosa'] else [1e-3,5e-3,1e-2])[lr_i]
    wd,drop,stop=reg[reg_i]
    r=dict(id=f'tune_{task}_{method}_c{lr_i*4+reg_i:02d}_s{seed}',phase='tune',task=task,method=method,seed=seed,config_id=lr_i*4+reg_i,
           adapter_lr=lr,head_lr=[2e-4,1e-3,5e-3][(lr_i+reg_i)%3],weight_decay=wd,dropout=drop,stopping=stop)
    path=S/(r['id']+'.json');path.write_text(json.dumps(r,indent=2)+'\n');rows.append(str(path))
(P/'tune_manifest.json').write_text(json.dumps(rows,indent=2)+'\n')
smokes=[]
for method in ['lora','unilora','prolosa','rosa']:
    r=json.loads((S/f'tune_mrpc_{method}_c04_s101.json').read_text());r.update(id='smoke_'+method,smoke=True)
    path=S/(r['id']+'.json');path.write_text(json.dumps(r,indent=2)+'\n');smokes.append(str(path))
(P/'smoke_manifest.json').write_text(json.dumps(smokes,indent=2)+'\n')
print(len(rows),'tuning runs')
