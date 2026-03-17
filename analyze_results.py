import os
import json
import glob
import pandas as pd

root_dir = "NLU/peft/examples/sequence_classification/results_glue_variants_unilora_block_routing_8_blocks/roberta-large/mrpc"
files = glob.glob(os.path.join(root_dir, "**/*.json"), recursive=True)

data = []
for file in files:
    try:
        with open(file, 'r') as f:
            content = json.load(f)
            
        args = content.get("args", {})
        best_metric = content.get("best_metric", {})
        
        lr = args.get("head_lr")
        seed = args.get("seed")
        acc = best_metric.get("accuracy")
        f1 = best_metric.get("f1")
        
        data.append({
            "LR": lr,
            "Seed": seed,
            "Accuracy": acc,
            "F1": f1,
            "File": os.path.basename(file)
        })
    except Exception as e:
        print(f"Error reading {file}: {e}")

df = pd.DataFrame(data)
if not df.empty:
    df = df.sort_values(by=["Seed", "LR"])
    print(df.to_string(index=False))
else:
    print("No data found.")
