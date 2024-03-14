import time
import numpy as np
import pandas as pd
import os
import argparse
from tqdm import tqdm
from pprint import pprint
from datetime import datetime
# import make_material
from make_material import sample_vignette
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer
)
from random import shuffle
from llama_logprobs import(
    getLogProbContinuation,
    use_jenns_method
)

# get device count
num_devices = torch.cuda.device_count()
print("NUM DEVICES: ", num_devices)
# print all device names and info
for i in range(num_devices):
    print(torch.cuda.get_device_properties(i))

def main(
    model_name, 
    task='ref_game',
    computation='use_own_scoring',
):
    date_out = datetime.now().strftime("%Y%m%d_%H%M")
    name_for_saving = model_name.split('/')[-1]

    # load model and tokenizer for Llama
    tokenizer = AutoTokenizer.from_pretrained(model_name, is_fast=False)
    if (computation == "use_own_scoring") or (computation == "use_jenns_method"):
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map='auto', 
            torch_dtype=torch.float16
        )
        model.eval()
    else:
        raise ValueError("Computation method not recognized. Please use 'use_own_scoring' or 'use_surprisal'.")

    list_of_dicts = []

    vignettes = pd.read_csv("../02-data/super_glue_formatted_boolq.csv")
    options = ["yes", "no"]
    
    if "chat" in model_name:
        input_template = "[INST]Context: {context}\nQuestion: {question}\n{instruction}\I would choose the answer [/INST]"
    else:
        input_template = "Context: {context}\nQuestion: {question}\n{instruction}\n\nYour answer:\n\nI would choose the answer "

    for i, row in vignettes.iterrows():
        # format instructions
        # shuffle answer choices 
        shuffle(options)
        instruction = "Which of the following answers would you choose:\n\n" + "\n".join(options) 
        print("Instruction ", instruction)
        # format the input
        prompt = input_template.format(
            context=row["sentence1"],
            question=row["sentence2"],
            instruction=instruction,
        )
        print(prompt)
        # retrieve log probabilities of correct and incorrect continuations
        predictions_correct = getLogProbContinuation(
            prompt, 
            '"' + row["correct"].lower() + '"',
            model,
            tokenizer,
        )
        print("#### Predictions correct: ", predictions_correct)
        predictions_incorrect = predictions_correct = getLogProbContinuation(
            prompt, 
            '"' + row["incorrect"].lower() + '"',
            model,
            tokenizer,
        )
        print("#### Predictions incorrect: ", predictions_incorrect)

        # for testing, also just sample a few productions
        predictions_prompt_ids = tokenizer(prompt.strip(), return_tensors="pt").to("cuda:0")
        production_samples = model.generate(
            **predictions_prompt_ids,
            max_new_tokens=16,
        )
        production_decoded = tokenizer.batch_decode(production_samples)

        results = {
            "scores_target": predictions_correct[0],
            "scores_target_npnlg": predictions_correct[1],
            "generation_correct": predictions_correct[2],
            "generation_score_correct": predictions_correct[3],
            "scores_distractor": predictions_incorrect[0],
            "scores_distractor_npnlg": predictions_incorrect[1],
            "generation_distractor": predictions_incorrect[2],
            "generation_score_distractor": predictions_incorrect[3],
            "production_decoded": "\n".join(production_decoded),
        }
        materials = {
            "trial": i,
            "context": row["sentence1"],
            "question": row["sentence2"],
            "correct": row["correct"],
            "incorrect": row["incorrect"],
        }
        output = dict(**materials, **results)
        list_of_dicts.append(output)

        results_df = pd.DataFrame(list_of_dicts)
        # continuous saving of results
        results_name = f'boolq_ownMeanScores_greedySamples_wQuots_{computation}_{name_for_saving}_{date_out}.csv'
        results_df.to_csv(results_name, index = False)