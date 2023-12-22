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
    prompt = preface + initialSequence + continuation
    print(initialSequence)
    print(prompt)
    # tokenize separately, so as to know the shape of the continuation
    input_ids_prompt = tokenizer(
        initialSequence.strip(), 
        return_tensors="pt",
        # add_special_tokens=False
    ).input_ids
    input_ids = tokenizer(
        prompt.strip(),
        return_tensors="pt",
    ).input_ids.to("cuda:0")
    
    print("input_ids prompt ",input_ids_prompt.shape, input_ids_prompt)
   
    print("input ids shape ", input_ids.shape, input_ids)

    # pass through model
    with torch.no_grad():
        outputs = model(
            input_ids,
            labels=input_ids
        )
    # transform logits to probabilities
    print("shape of logits ", outputs.logits.shape)
    # remove the EOS logit which we aren't interested in
    llama_output_scores = logsoftmax(
        outputs.logits[0][:-1]
    )
    print("output log probs shape ", llama_output_scores.shape)
    # retreive log probs at token ids
    # transform input_ids to a tensor of shape [n_tokens, 1] for this
    # cut off the sos token so as to get predictions for the actual token conditioned on 
    # preceding context
    input_ids_probs = input_ids[:, 1:].squeeze().unsqueeze(-1)
    #print("shape of input ids for prob retrieval ", input_ids_probs.shape, input_ids_probs)
    # retreive at correct token positions
    conditionalLogProbs = torch.gather(
        llama_output_scores, 
        dim=-1, 
        index=input_ids_probs
    ).flatten()
    # slice output to only get scores of the continuation, not the context
    continuationConditionalLogProbs = conditionalLogProbs[
        (input_ids_prompt.shape[-1]-1):
    ]
    print("Shape of retrieved log probs", continuationConditionalLogProbs.shape, continuationConditionalLogProbs)
    # compute continunation log prob
    sentLogProb = torch.sum(continuationConditionalLogProbs).item()
    print("sent log prob ", sentLogProb)
    print("mean log prob ", torch.mean(continuationConditionalLogProbs).item())

    ### alternative method of retrieving log probs of single words via generate ###
    # only pass the prompt and then retreive score of the respective tokens among the first predicted token
    outputs_generate = model.generate(
        input_ids_prompt.to("cuda:0"), 
        max_new_tokens=5,
        output_scores=True,
        num_return_sequences=1,
        return_dict_in_generate=True
    )
    if isinstance(outputs_generate.scores, tuple):
        logits = outputs_generate.scores[0][0]
    else:
        logits = outputs_generate.scores

    input_ids_continuation = input_ids[0][input_ids_prompt.shape[-1]:]
    print("indices of nonzero generation scores ", (logits > -torch.inf).nonzero())        
    answer_logits = logits[input_ids_continuation[0]].item()
    print("input_ids_continuation[0][-1] ", input_ids_continuation[0])
    print("outputs generate scores shape ", len(outputs_generate.scores), outputs_generate.scores[0][0].shape, outputs_generate.scores[0].shape)
    print("Answer logit retrieved with Jenn's method ", answer_logits)
    #print("Token predicted with 1 token generate ", outputs_generate.sequences, tokenizer.batch_decode(outputs_generate.sequences))
    ### sanity checking the llh results via nll loss comp
    manual_llh = torch.mean(conditionalLogProbs)
    auto_llh = model(
       input_ids,
       labels=input_ids
    ).loss
    print("manually computed LL ", manual_llh)
    print("loss computation based nll ", auto_llh)

    # another sanity check with MF's NPNLG code
    relevant_labels = torch.clone(input_ids)
    for i in range(input_ids_prompt.shape[-1]):
        relevant_labels[0, i] = -100
    print("Relevant labels ", relevant_labels)
    output_masked = model(input_ids, labels=relevant_labels)
    print("Output loss (i.e., mean) computed with NPNLG approach ", output_masked.loss.item(), output_masked.loss.item() * (input_ids_continuation.shape[-1]))
    # for doubke checking, compute the same with only the last tokens
    print("input ids continuation for checking NPNLG code" , input_ids_continuation)
    output_last_tokens_loss = model(input_ids_continuation.unsqueeze(0), labels=input_ids_continuation.unsqueeze(0))
    print("npnlg double checking loss ", output_last_tokens_loss.loss.item())

    return sentLogProb, output_masked.loss.item()
            

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
    # general sanity check
    #testing_prompt = getLogProbContinuation(
    #    "Continue the following Christmas song: Dashing through the snow\nIn a one-horse open sleigh\nO'er the fields we go\nLaughing all the way\nBells on bobtails ring\nMaking spirits bright\nWhat fun it is to ride and sing\nA sleighing song tonight\nJingle bells, Jingle bells\n", "Jingle",
    #    model, tokenizer)
    #print("Jingle continuation " ,testing_prompt) 
    #testing_prompt2 = getLogProbContinuation(
    #    "Continue the following Christmas song: Dashing through the snow\nIn a one-horse open sleigh\nO'er the fields we go\nLaughing all the way\nBells on bobtails ring\nMaking spirits bright\nWhat fun it is to ride and sing\nA sleighing song tonight\nJingle bells, Jingle bells\n", "Christmas",
    #    model, tokenizer)
    #print("Christmas continuation ", testing_prompt2)

    # production
    lprob_target, lprob_target_gen      = getLogProbContinuation(
        context_production, '"' + vignette["production_target"] + '"',
        model, tokenizer)
    lprob_competitor, lprob_competitor_gen  = getLogProbContinuation(
        context_production, '"' + vignette["production_competitor"] + '"',
        model, tokenizer)
    lprob_distractor1, lprob_distractor1_gen = getLogProbContinuation(
        context_production, '"' + vignette["production_distractor1"] + '"',
        model, tokenizer)
    lprob_distractor2, lprob_distractor2_gen = getLogProbContinuation(
        context_production, '"' + vignette["production_distractor2"] + '"',
        model, tokenizer)
    # for testing, also just sample a few productions
    # predictions_prompt_ids = tokenizer(context_production, return_tensors="pt")
    # production_samples = model.generate(
    #     **predictions_prompt_ids,
    #     do_sample = True,
    #     num_return_sequences=5,
    # )
    # production_decoded = tokenizer.batch_decode(production_samples)
    # print("productions decoded", production_decoded)
    scores_production = np.array([lprob_target, lprob_competitor, lprob_distractor1, lprob_distractor2])
    probs_production = soft_max(scores_production, alpha=alpha_production)
    # softmax the scores generated with alternative method
    scores_production_gen = np.array([lprob_target_gen, lprob_competitor_gen, lprob_distractor1_gen, lprob_distractor2_gen])
    probs_production_gen = soft_max(scores_production_gen, alpha=alpha_interpretation)
    # interpretation

    lprob_target, lprob_target_gen      = getLogProbContinuation(
        context_interpretation, '"' + vignette["interpretation_target"] + '"',
        model, tokenizer)
    lprob_competitor, lprob_comp_gen  = getLogProbContinuation(
        context_interpretation, '"' +  vignette["interpretation_competitor"] + '"' ,
        model, tokenizer)
    lprob_distractor, lprob_distractor_gen  = getLogProbContinuation(
        context_interpretation, '"'+ vignette["interpretation_distractor"]  + '"',
        model, tokenizer)

    scores_interpretation = np.array([lprob_target, lprob_competitor, lprob_distractor])
    probs_interpretation = soft_max(scores_interpretation, alpha=1)

    scores_interpretation_gen = np.array([lprob_target_gen, lprob_comp_gen, lprob_distractor_gen])
    probs_interpretation_gen = soft_max(scores_interpretation_gen, alpha=1
                                    )

    output_dict = {
        'alpha_production'              : alpha_production,
        'scores_production_target'      : scores_production[0],
        'scores_production_competitor'  : scores_production[1],
        'scores_production_distractor1' : scores_production[2],
        'scores_production_distractor2' : scores_production[3],

        'scores_production_target_npnlg'      : scores_production_gen[0],
        'scores_production_competitor_npnlg'  : scores_production_gen[1],
        'scores_production_distractor1_npnlg' : scores_production_gen[2],
        'scores_production_distractor2_npnlg' : scores_production_gen[3],

        'prob_production_target'        : probs_production[0],
        'prob_production_competitor'    : probs_production[1],
        'prob_production_distractor1'   : probs_production[2],
        'prob_production_distractor2'   : probs_production[3],
        'prob_production_target_npnlg'        : probs_production_gen[0],
        'prob_production_competitor_npnlg'    : probs_production_gen[1],
        'prob_production_distractor1_npnlg'   : probs_production_gen[2],
        'prob_production_distractor2_npnlg'   : probs_production_gen[3],
        # 'production_decoded': "|".join(production_decoded),
        'alpha_interpretation'             : alpha_interpretation,
        'scores_interpretation_target'     : scores_interpretation[0],
        'scores_interpretation_competitor' : scores_interpretation[1],
        'scores_interpretation_distractor' : scores_interpretation[2],
        'prob_interpretation_target'       : probs_interpretation[0],
        'prob_interpretation_competitor'   : probs_interpretation[1],
        'prob_interpretation_distractor'   : probs_interpretation[2],
        'scores_interpretation_target_npnlg'     : scores_interpretation_gen[0],
        'scores_interpretation_competitor_npnlg' : scores_interpretation_gen[1],
        'scores_interpretation_distractor_npnlg' : scores_interpretation_gen[2],
        'prob_interpretation_target_npnlg'       : probs_interpretation_gen[0],
        'prob_interpretation_competitor_npnlg'   : probs_interpretation_gen[1],
        'prob_interpretation_distractor_npnlg'   : probs_interpretation_gen[2]
    }

    return output_dict


def main(model_name):

    # load model and tokenizer for Llama
    tokenizer = AutoTokenizer.from_pretrained(model_name, is_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map='auto', 
        torch_dtype=torch.float16
    )
    model.eval()

    list_of_dicts = []

    # for comparability of results, use materials from GPT-3 results
    vignettes = pd.read_csv('../02-data/results_GPT.csv')
    # vignettes = pd.read_csv('../02-data/sanity_check_data.csv')
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

    results_name = f'results_wQuots_{name_for_saving}.csv'
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
