# saves the openwebtext dataset to a binary file for training. following was helpful:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py

import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset, load_from_disk # huggingface datasets
from transformers import AutoTokenizer

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = 8

# takes 54GB in huggingface .cache dir, about 8M documents (8,013,769)
dataset = load_dataset("HuggingFaceH4/databricks_dolly_15k")

# owt by default only contains the 'train' split, so create a test split
split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
split_dataset['val'] = split_dataset.pop('test') # rename the test split to val

# we now want to tokenize the dataset. first define the encoding function (gpt2 bpe)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
def process(example):
    tokenized_output = tokenizer(example['text'], padding='max_length', truncation=True, add_special_tokens=False) # encode_ordinary ignores any special tokens 
    target_tokenized_output = tokenizer(example['target_text'], padding='max_length', truncation=True, add_special_tokens=True)
    tokenized_output['target'] = target_tokenized_output['input_ids']
    return tokenized_output

def process_text(example):
    instruction = example['instruction']
    input = example['input']
    output = example['output']
    if input != '':
        text = f"User:{instruction}\nContext:{input}\nAssistant:{output}"
    else:
        text = f"User:{instruction}\nAssistant:{output}"
    target_text = f"{text}{tokenizer.eos_token}"
    example['text'] = text
    example['target_text'] = target_text
    return example
intermediate_tokenized = split_dataset.map(
process_text,
num_proc=num_proc
)
# batch passes in example as a _batch_ into process. So example['text'] is batch_size
tokenized = intermediate_tokenized.map(
    process,
    desc="tokenizing the splits",
    batched=True
)

tokenized.save_to_disk('databricks_dolly')

