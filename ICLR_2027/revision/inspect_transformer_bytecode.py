"""Read-only audit of retained CPython 3.11 code objects; never execute them."""
import dis
import marshal
from pathlib import Path
import sys
import types
assert sys.version_info[:2] == (3, 11), 'Use CPython 3.11 for the retained bytecode'
repo=Path(__file__).resolve().parents[2]
out=Path(__file__).resolve().parent/'existing_evidence/transformer_bytecode_audit.txt'
with out.open('w') as f:
    for name in ['runner','teacher','common']:
        path=repo/'synthetic/transformer_lab/__pycache__'/f'{name}.cpython-311.pyc'
        code=marshal.loads(path.read_bytes()[16:])
        print(path.relative_to(repo),'source=',code.co_filename,file=f)
        for item in code.co_consts:
            if isinstance(item,types.CodeType):
                print(item.co_name,item.co_firstlineno,item.co_varnames,file=f)
                if name=='runner' or item.co_name in ['plant_weight_update','attach_projection_stats','summarize_function_estimates']:
                    dis.dis(item,file=f)
print(out)
