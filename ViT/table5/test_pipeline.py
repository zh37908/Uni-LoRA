"""Check the expensive-run gate and test-set isolation without training."""
import json
import tempfile
import unittest
from pathlib import Path

import pipeline as q
import finish


class SelectionGateTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.old_root, self.old_state = q.ROOT, q.STATE
        q.ROOT = Path(self.temp.name)
        q.STATE = q.ROOT / "state.json"
        (q.ROOT / "specs").mkdir()
        q.initialize()
        self.state = json.loads(q.STATE.read_text())

    def tearDown(self):
        q.ROOT, q.STATE = self.old_root, self.old_state
        self.temp.cleanup()

    def finish(self, tasks, accuracy=0.8):
        for i,t in enumerate(tasks):
            path = q.ROOT / f"result_{q.signature(t['spec'])}.json"
            # Scores depend only on validation; tuning files contain no test metric.
            r = dict(spec_sha256=q.signature(t["spec"]), best_val_accuracy=accuracy+i*1e-5,
                     best_val_loss=1-accuracy)
            path.write_text(json.dumps(r))
            t.update(status="done", result=str(path))

    def test_no_final_until_every_group_has_two_seed_validation(self):
        self.finish(q.tasks_for(self.state,"smoke"))
        q.advance(self.state)
        self.assertEqual(len(q.tasks_for(self.state,"screen")),384)
        self.finish(q.tasks_for(self.state,"screen"))
        q.advance(self.state)
        self.assertEqual(len(q.tasks_for(self.state,"validate")),48)
        self.finish(q.tasks_for(self.state,"validate"))
        q.advance(self.state)
        confirmations = q.tasks_for(self.state,"confirm")
        self.assertEqual(len(confirmations),32)
        self.finish(confirmations[:-1])
        q.advance(self.state)
        self.assertEqual(q.tasks_for(self.state,"final"),[])
        self.finish(confirmations[-1:])
        q.advance(self.state)
        final = q.tasks_for(self.state,"final")
        self.assertEqual(len(final),80)
        for group in q.GROUPS:
            tasks = q.tasks_for(self.state,"final",group)
            self.assertEqual({t["spec"]["seed"] for t in tasks},{42,43,44,45,46})
            self.assertEqual({t["spec"]["epochs"] for t in tasks},{20})
            self.assertEqual(len({q.signature(q.configuration(t)) for t in tasks}),1)
        q.advance(self.state)
        self.assertEqual(len(q.tasks_for(self.state,"final")),80)

    def test_tuning_results_with_test_scores_are_rejected(self):
        t = q.tasks_for(self.state,"smoke")[0]
        self.finish([t])
        p = Path(t["result"])
        r = json.loads(p.read_text()); r["test_accuracy"] = 1.0
        p.write_text(json.dumps(r))
        with self.assertRaises(AssertionError): q.result(t)

    def test_final_export_requires_all_runs_and_uses_five_seed_statistics(self):
        with self.assertRaises(AssertionError): finish.export(self.state)
        cfg=q.configs()[0]
        for model,dataset in q.GROUPS:
            for seed in range(42,47):
                q.add(self.state,"final",model,dataset,cfg,seed,20)
        for t in q.tasks_for(self.state,"final"):
            spec=t["spec"]
            p=q.ROOT/f'final_{q.signature(spec)}.json'
            r=dict(spec=spec,spec_sha256=q.signature(spec),test_accuracy=0.8+0.01*(spec["seed"]-42),
                   best_val_accuracy=0.75,best_epoch=18,d=48000,K=24000)
            p.write_text(json.dumps(r)); t.update(status="done",result=str(p))
        finish.export(self.state)
        stats=json.loads((q.ROOT/"table5_statistics.json").read_text())
        self.assertAlmostEqual(stats["base/average"]["mean"],82.0)
        self.assertAlmostEqual(stats["large/cifar100"]["std"],1.5811388300841898)
        self.assertEqual(len((q.ROOT/"final_runs.csv").read_text().splitlines()),81)


if __name__ == "__main__":
    unittest.main()
