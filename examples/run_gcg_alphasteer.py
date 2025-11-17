"""
# GCG use case example.

# Using the GCG algorithm to optimize the prompt for the vicuna model to output the desired content.
"""
import easyjailbreak
import torch
import logging
from transformers import set_seed
from easyjailbreak.datasets import JailbreakDataset
from transformers import AutoConfig

import os
import datetime
now = datetime.datetime.now()
checkpoint_dir = f'./exp/{now.strftime("%Y%m%d")}/{now.strftime("%H%M%S")}'
os.environ['CHECKPOINT_DIR'] = checkpoint_dir

logging.basicConfig(
    level=logging.DEBUG, 
    format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s'

)
set_seed(42)

# Load model and tokenizer
# Load model and tokenizer
model = easyjailbreak.models.from_pretrained_alphasteer(
    model_name_or_path='meta-llama/Llama-3.1-8B-Instruct', 
    dtype=torch.bfloat16, 
    max_new_tokens=40
)

steering_matrix = torch.load(
    '/data/shenc/projects/AlphaSteer/data/steering_matrix/steering_matrix_llama3.1.pt',
    dtype=torch.bfloat16,
    map_location='cuda:0'
    )
steering_layers = [8, 9, 10, 11, 12, 13, 14, 16, 18, 19]
steering_strength = [0.0] * model.config.num_hidden_layers
for layer in steering_layers:
    steering_strength[layer] = -0.5

model.set_steering_parameters(
    steering_matrix=steering_matrix,
    strength=steering_strength
)

# Load dataset
# dataset_name = 'AdvBench'
# dataset = JailbreakDataset(dataset_name)
dataset = JailbreakDataset(
    "advbench_test.json",
    shuffle=True,
    local_file_type="json")


# API class initialization and settings
attacker = easyjailbreak.attacker.GCG(
    attack_model=model,
    target_model=model,
    jailbreak_datasets=dataset,
    jailbreak_prompt_length=20,
    num_turb_sample=512,
    batchsize=256,      # decrease if OOM
    top_k=256,
    max_num_iter=500
)

# Launch the attack
attacker.attack()

attacker.jailbreak_datasets.save_to_jsonl(f'gcg_result.jsonl')