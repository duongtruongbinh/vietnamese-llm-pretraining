#!/usr/bin/env python3
"""
Perplexity calculator from trainer_state.json checkpoint logs.
"""

import json
import math

CHECKPOINT_PATH = './vietnamese_gpt2_pre_en_mixed_data/checkpoint-3000/trainer_state.json'

with open(CHECKPOINT_PATH) as f:
    data = json.load(f)
    for entry in data['log_history']:
        if 'eval_loss' in entry:
            perplexity = math.exp(entry['eval_loss'])
            print(f"Step {entry['step']}: Perplexity = {perplexity:.2f}")
