import time
import numpy as np
import pandas as pd
import os
import argparse
from tqdm import tqdm
from pprint import pprint
# import make_material
from make_material import sample_vignette
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer
)
# get device count
num_devices = torch.cuda.device_count()
print("NUM DEVICES: ", num_devices)
# print all device names and info
for i in range(num_devices):
    print(torch.cuda.get_device_properties(i))

# define logsoftmax for retrieving logprobs from scores
logsoftmax = torch.nn.LogSoftmax(dim=-1)

def getLogProbContinuation(
        initialSequence, 
        continuation, 
        model,
        tokenizer,
        preface = ''):
    """
    Helper for retrieving log probability of different response types from Llama-2 of various sizes.
    """

    initialSequence = preface + initialSequence
    # tokenize separately, so as to know the shape of the continuation
    input_ids_prompt = tokenizer(
        initialSequence, 
        return_tensors="pt",
    ).input_ids
    input_ids_continuation = tokenizer(
        continuation,
        return_tensors="pt",
    ).input_ids
    #print("input_ids prompt ", input_ids_prompt)
    #print("input ids continuation ", input_ids_continuation)
    # cut off the first token of the continuation, as it is SOS
    input_ids = torch.cat(
        (input_ids_prompt, input_ids_continuation[:, 1:]), 
        -1
    ).to("cuda:0") # put input on the first device
    # pass through model
    outputs = model(
        input_ids,
    )
    # transform logits to probabilities
    llama_output_scores = logsoftmax(
        outputs.logits[0][:, :-1]
    )
    # retreive log probs at token ids
    # transform input_ids to a tensor of shape [n_tokens, 1] for this
    # cut off the sos token so as to get predictions for the actual token conditioned on 
    # preceding context
    input_ids_probs = input_ids[:, 1:].squeeze().unsqueeze(-1)
    # retreive at correct token positions
    conditionalLogProbs = torch.gather(
        llama_output_scores, 
        dim=-1, 
        index=input_ids_probs
    ).flatten()
    # slice output to only get scores of the continuation, not the context
    continuationConditionalLogProbs = conditionalLogProbs[
        input_ids_prompt.shape[-1]:
    ]
    print("Shape of retrieved log probs", continuationConditionalLogProbs.shape)
    # compute continunation log prob
    sentLogProb = torch.sum(continuationConditionalLogProbs).item()
    print("sent log prob ", sentLogProb)

    return sentLogProb
            

def soft_max(scores, alpha=1):
    scores = np.array(scores)
    output = np.exp(scores * alpha)
    return(output / np.sum(output))


def get_model_predictions(
        vignette, 
        model,
        tokenizer,
        model_name,
        alpha_production, 
        alpha_interpretation
    ):

    # take care of special tokens for chat models
    # assume that the task and context come from the user, and the response from the model
    # no specific system prompt is passed
    # if one wanted to, the expected formatting would be: [INST]<<SYS>>{system prompt}<</SYS>>\n\n{user message}[/INST]
    if "chat" in model_name:
        context_production = f"[INST]{vignette['context_production']}[/INST]"
        context_interpretation = f"[INST]{vignette['context_interpretation']}[/INST]"
    else:
        context_production = vignette['context_production']
        context_interpretation = vignette['context_interpretation']

    # production
    lprob_target      = getLogProbContinuation(
        context_production, vignette["production_target"],
        model, tokenizer)
    lprob_competitor  = getLogProbContinuation(
        context_production, vignette["production_competitor"],
        model, tokenizer)
    lprob_distractor1 = getLogProbContinuation(
        context_production, vignette["production_distractor1"],
        model, tokenizer)
    lprob_distractor2 = getLogProbContinuation(
        context_production, vignette["production_distractor2"],
        model, tokenizer)

    scores_production = np.array([lprob_target, lprob_competitor, lprob_distractor1, lprob_distractor2])
    probs_production = soft_max(scores_production, alpha_production)

    # interpretation

    lprob_target      = getLogProbContinuation(
        context_interpretation, vignette["interpretation_target"],
        model, tokenizer)
    lprob_competitor  = getLogProbContinuation(
        context_interpretation, vignette["interpretation_competitor"],
        model, tokenizer)
    lprob_distractor  = getLogProbContinuation(
        context_interpretation, vignette["interpretation_distractor"],
        model, tokenizer)

    scores_interpretation = np.array([lprob_target, lprob_competitor, lprob_distractor])
    probs_interpretation = soft_max(scores_interpretation, alpha_interpretation)

    output_dict = {
        'alpha_production'              : alpha_production,
        'scores_production_target'      : scores_production[0],
        'scores_production_competitor'  : scores_production[1],
        'scores_production_distractor1' : scores_production[2],
        'scores_production_distractor2' : scores_production[3],
        'prob_production_target'        : probs_production[0],
        'prob_production_competitor'    : probs_production[1],
        'prob_production_distractor1'   : probs_production[2],
        'prob_production_distractor2'   : probs_production[3],
        'alpha_interpretation'             : alpha_interpretation,
        'scores_interpretation_target'     : scores_interpretation[0],
        'scores_interpretation_competitor' : scores_interpretation[1],
        'scores_interpretation_distractor' : scores_interpretation[2],
        'prob_interpretation_target'       : probs_interpretation[0],
        'prob_interpretation_competitor'   : probs_interpretation[1],
        'prob_interpretation_distractor'   : probs_interpretation[2]

    }

    return output_dict


def main(model_name):

    # load model and tokenizer for Llama
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map='auto', 
        torch_dtype=torch.float16
    )
    model.eval()

    list_of_dicts = []

    # for comparability of results, use materials from GPT-3 results
    vignettes = pd.read_csv('results.csv')

    for i, vignette in tqdm(vignettes.iterrows()):
        predictions = get_model_predictions(
            vignette, 
            model, 
            tokenizer, 
            model_name,
            0.5, 
            0.5
        )

        materials = {
            'trial': i,
            'utterances': vignette['utterances'],
            'trigger_feature': vignette['trigger_feature'],	
            'nuisance_feature': vignette['nuisance_feature'],	
            'production_target': vignette['production_target'],	
            'production_competitor': vignette['production_competitor'],	
            'production_distractor1': vignette['production_distractor1'],
            'production_distractor2': vignette['production_distractor2'],	
            'production_index_target': vignette['production_index_target'],	
            'production_index_competitor': vignette['production_index_competitor'],	
            'production_index_distractor1': vignette['production_index_distractor1'],	
            'production_index_distractor2': vignette['production_index_distractor2'],
            'trigger_object': vignette['trigger_object'],	
            'trigger_word': vignette['trigger_word'],	
            'interpretation_target': vignette['interpretation_target'],	
            'interpretation_competitor': vignette['interpretation_competitor'],	
            'interpretation_distractor': vignette['interpretation_distractor'],	
            'interpretation_index_target': vignette['interpretation_index_target'],	
            'interpretation_index_competitor': vignette['interpretation_index_competitor'],	
            'interpretation_index_distractor': vignette['interpretation_index_distractor'],	
            'context_production': vignette['context_production'],	
            'context_interpretation': vignette['context_interpretation']
            
        }
        output = dict(**materials, **predictions)
        list_of_dicts.append(output)

    results_df = pd.DataFrame(list_of_dicts)

    pprint(results_df)
    # TODO format results_df name to include model name
    name_for_saving = model_name.split('/')[-1]
    results_name = f'results_{name_for_saving}.csv'
    results_df.to_csv(results_name, index = False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="meta-llama/Llama-2-7b-hf", 
        help="Model name"
    )
    args = parser.parse_args()

    main(args.model_name)
