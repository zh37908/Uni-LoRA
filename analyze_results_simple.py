import os
import json
import glob

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
        
        if acc is not None:
            data.append({
                "LR": lr,
                "Seed": seed,
                "Accuracy": acc,
                "F1": f1
            })
    except Exception as e:
        print(f"Error reading {file}: {e}")

# Sort by Seed then LR
data.sort(key=lambda x: (x["Seed"], x["LR"]))

print(f"{'Seed':<5} {'LR':<10} {'Accuracy':<10} {'F1':<10}")
print("-" * 35)
for row in data:
    print(f"{row['Seed']:<5} {row['LR']:<10} {row['Accuracy']:.4f}     {row['F1']:.4f}")
