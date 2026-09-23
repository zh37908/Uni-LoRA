"""Validate packaged protocol, original core-function parity and recorded statistics."""
import ast,csv,hashlib,json,math
from common import ROOT,PREPARED,sha256
from summarize import summarize,GROUPS
from data_utils import PromptFilter

def main():
    provenance=json.loads((ROOT/'results/provenance.json').read_text())
    for name,expected in provenance['core_function_ast_sha256'].items():
        actual={n.name:hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest() for n in ast.parse((ROOT/name).read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        for func,digest in expected.items():assert actual[func]==digest,(name,func)
    configs=[json.loads(f.read_text()) for f in (ROOT/'configs').glob('*.json')]
    assert len(configs)==12
    for group in GROUPS:
        rows=[c for c in configs if c['search_id'].startswith(group+'_r4_seed')]
        assert sorted(c['seed'] for c in rows)==[0,21,42]
        for c in rows:
            assert c['rank']==4 and c['expected_D']==10485760 and c['snip_reduction']=='max'
            assert c['budget']==(10485760 if group=='lora' else 2621440)
            assert c['d']+c['k']==c['budget'] and math.ceil(c['k']/c['expected_D']*c['expected_D'])==c['k']
    with (ROOT/'results/three_seeds.csv').open() as f:rows=list(csv.DictReader(f))
    actual=summarize(rows);expected=json.loads((ROOT/'results/summary.json').read_text())['groups']
    for group in GROUPS:
        for key,bench,metric in [('ifeval_prompt_strict','ifeval','prompt_strict'),('ifbench_prompt_loose','ifbench','prompt_loose')]:
            for value in ['mean_percent','sample_sd_pp','new_seeds_0_21_mean_percent']:
                assert math.isclose(actual[group][key][value],expected[group][bench][metric][value],abs_tol=1e-12)
    try:summarize(rows[:-1])
    except AssertionError:pass
    else:raise AssertionError('Missing seed accepted')
    checker=PromptFilter(['Keep exactly three items in this simple list today'])
    assert checker.overlaps('KEEP exactly three items in this simple list today!')
    assert not checker.overlaps('A completely unrelated topic')
    # If corpus preparation has run, verify raw splits against the experiment.
    if (PREPARED/'manifest.json').exists():
        expected=json.loads((ROOT/'expected_data.json').read_text())
        for split,info in expected['splits'].items():assert sha256(PREPARED/f'{split}.jsonl')==info['raw_sha256']
    print('PACKAGE_TESTS_PASSED: original core AST, 12 configs, exact budget, three-seed statistics, missing-seed rejection, prompt filter')

if __name__=='__main__':main()
