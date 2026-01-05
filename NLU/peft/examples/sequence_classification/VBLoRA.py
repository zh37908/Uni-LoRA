#!/usr/bin/env python
# coding: utf-8

# # Using VB-LoRA for sequence classification

# In this example, we fine-tune Roberta on a sequence classification task using VB-LoRA.
# 
# This notebook is adapted from `examples/sequence_classification/VeRA.ipynb`.

# ## Imports

# In[ ]:


import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from peft import (
    get_peft_model,
    VBLoRAConfig,
    PeftType,
)

import evaluate
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm


# ## Parameters

# In[ ]:


batch_size = 32
model_name_or_path = "roberta-large"
task = "mrpc"
peft_type = PeftType.VBLORA
device = torch.accelerator.current_accelerator().type if hasattr(torch, "accelerator") else "cuda"
num_epochs = 20
rank = 4
max_length = 128
num_vectors = 90
vector_length = 256
torch.manual_seed(0)


# In[3]:


peft_config = VBLoRAConfig(
    task_type="SEQ_CLS", 
    r=rank,
    topk=2,
    target_modules=['key', 'value', 'query', 'output.dense', 'intermediate.dense'],
    num_vectors=num_vectors,
    vector_length=vector_length,
    save_only_topk_weights=True, # Set to True to reduce storage space. Note that the saved parameters cannot be used to resume training from checkpoints.
    vblora_dropout=0.,
)
head_lr = 4e-3
vector_bank_lr = 1e-3
logits_lr = 1e-2


# ## Loading data

# In[4]:


if any(k in model_name_or_path for k in ("gpt", "opt", "bloom")):
    padding_side = "left"
else:
    padding_side = "right"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side=padding_side)
if getattr(tokenizer, "pad_token_id") is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id


# In[5]:


datasets = load_dataset("glue", task)
metric = evaluate.load("glue", task)


# In[6]:


def tokenize_function(examples):
    # max_length=None => use the model max length (it's actually the default)
    outputs = tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, max_length=max_length)
    return outputs


tokenized_datasets = datasets.map(
    tokenize_function,
    batched=True,
    remove_columns=["idx", "sentence1", "sentence2"],
)

# We also rename the 'label' column to 'labels' which is the expected name for labels by the models of the
# transformers library
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")


# In[7]:


def collate_fn(examples):
    return tokenizer.pad(examples, padding="longest", return_tensors="pt")


# Instantiate dataloaders.
train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, collate_fn=collate_fn, batch_size=batch_size)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], shuffle=False, collate_fn=collate_fn, batch_size=batch_size
)


# ## Preparing the VB-LoRA model

# In[8]:


model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, return_dict=True, max_length=None)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
model.print_savable_parameters()


# In[9]:


from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names

decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
decay_parameters = [name for name in decay_parameters if "bias" not in name]
vector_bank_parameters = [name for name, _ in model.named_parameters() if "vector_bank" in name]
logits_parameters = [name for name, _ in model.named_parameters() if "logits" in name ]

optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if n in decay_parameters and \
                    n not in logits_parameters and n not in vector_bank_parameters],
        "weight_decay": 0.1,
        "lr": head_lr,
    },
    {
        "params": [p for n, p in model.named_parameters() if n not in decay_parameters and \
                    n not in logits_parameters and n not in vector_bank_parameters],
        "weight_decay": 0.0,
        "lr": head_lr,
    },
    {
        "params": [p for n, p in model.named_parameters() if n in vector_bank_parameters],
        "lr": vector_bank_lr,
        "weight_decay": 0.0,
    },
    {
        "params": [p for n, p in model.named_parameters() if n in logits_parameters],
        "lr": logits_lr,
        "weight_decay": 0.0,
    },
]

optimizer = AdamW(optimizer_grouped_parameters)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0.06 * (len(train_dataloader) * num_epochs),
    num_training_steps=(len(train_dataloader) * num_epochs),
)


# ## Training

# In[10]:


model.to(device)

for epoch in range(num_epochs):
    model.train()
    for step, batch in enumerate(tqdm(train_dataloader)):
        batch.to(device)
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    model.eval()
    for step, batch in enumerate(tqdm(eval_dataloader)):
        batch.to(device)
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        predictions, references = predictions, batch["labels"]
        metric.add_batch(
            predictions=predictions,
            references=references,
        )

    eval_metric = metric.compute()
    print(f"epoch {epoch}:", eval_metric)


# ## Share adapters on the 🤗 Hub

# In[11]:


account_id = ...  # your Hugging Face Hub account ID


# In[ ]:


model.push_to_hub(f"{account_id}/roberta-large-peft-vblora")


# ## Load adapters from the Hub
# 
# You can also directly load adapters from the Hub using the commands below:

# In[13]:


import torch
from peft import PeftModel, PeftConfig
from transformers import AutoTokenizer


# In[14]:


peft_model_id = f"{account_id}/roberta-large-peft-vblora"
config = PeftConfig.from_pretrained(peft_model_id)
inference_model = AutoModelForSequenceClassification.from_pretrained(config.base_model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)


# In[15]:


# Load the model
inference_model = PeftModel.from_pretrained(inference_model, peft_model_id)


# In[16]:


inference_model.to(device)
inference_model.eval()
for step, batch in enumerate(tqdm(eval_dataloader)):
    batch.to(device)
    with torch.no_grad():
        outputs = inference_model(**batch)
    predictions = outputs.logits.argmax(dim=-1)
    predictions, references = predictions, batch["labels"]
    metric.add_batch(
        predictions=predictions,
        references=references,
    )

eval_metric = metric.compute()
print(eval_metric)

