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
from surprisal import AutoHuggingFaceModel


# get device count
num_devices = torch.cuda.device_count()
print("NUM DEVICES: ", num_devices)
# print all device names and info
for i in range(num_devices):
    print(torch.cuda.get_device_properties(i))

# define logsoftmax for retrieving logprobs from scores
logsoftmax = torch.nn.LogSoftmax(dim=-1)

TOKENS_MAP = {
    '"triangle"': 3,
    '"circle"': 3,
    '"hexagon"': 4,
    '"square"': 3,
    '"red"': 3,
    '"blue"': 3,
    '"green"': 3,
    '"orange"': 4,
    '"stripes"': 5,
    '"spades"': 4,
    '"dots"': 3,
    '"stars"': 4
}

def get_relevant_generated_tokens(
    generated_sequence,
    tokens_map=TOKENS_MAP,
):
    for token, num in tokens_map.items():
        if token in generated_sequence:
            return num

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
    print("len cond log P ", len(conditionalLogProbs), input_ids_prompt.shape[-1]-1)
    print("Shape of retrieved log probs", continuationConditionalLogProbs.shape, continuationConditionalLogProbs)
    # compute continunation log prob
    sentLogProb = torch.sum(continuationConditionalLogProbs).item()
    meanLogProb = torch.mean(continuationConditionalLogProbs).item()
    print("sent log prob ", sentLogProb)
    print("mean log prob ", torch.mean(continuationConditionalLogProbs).item())

    ###### production-task specific exploration #######
    ### alternative method of retrieving log probs of single words via generate ###
    # only pass the prompt and then retreive score of the respective tokens among the first predicted token
    outputs_generate = model.generate(
        input_ids_prompt.to("cuda:0"),
        do_sample=False, 
        max_new_tokens=5,
        output_scores=True,
        num_return_sequences=1,
        return_dict_in_generate=True
    )
    if isinstance(outputs_generate.scores, tuple):
        print("Using first method of retrieving logits")
        logits = outputs_generate.scores #[0][0]
    else:
        print("Using second method of retrieving logits")
        logits = outputs_generate.scores
    generate_logprobs = logsoftmax(torch.stack(logits)).squeeze()
    print("stack shape ", torch.stack(logits).shape)
    print("generation log probs shape", generate_logprobs.shape)
    # print("Logits shape ", logits.shape)
    print("Outputs generate sequences ", outputs_generate.sequences)
    first_generated_sequence = tokenizer.decode(outputs_generate.sequences[0])
    print("First generated sequence ", first_generated_sequence)
    input_ids_continuation = input_ids[0][input_ids_prompt.shape[-1]:]
    print("input ids continuation shape ", input_ids_continuation.shape)
    generated_continuation = tokenizer.decode(
        outputs_generate.sequences[0][input_ids_prompt.shape[-1]:]
    )
    print("Generated continuation ", generated_continuation)
    # grab the first word's log prob (because that is one of the options in quotes)
    # via poor man's mapping: we check if the tokens of each of the options are in the continuation
    # and grad the respective log probs
    relevant_tokens_num = get_relevant_generated_tokens(
        generated_continuation,
        tokens_map = TOKENS_MAP,
    )
    if relevant_tokens_num is not None:
        relevant_tokens = outputs_generate.sequences[0][
            input_ids_prompt.shape[-1]:(input_ids_prompt.shape[-1]+relevant_tokens_num)
        ]
        print("relevant token nums ", relevant_tokens_num)
        print("relevant tokens ", relevant_tokens)
        # get their log P
        relevant_word_log_probs = []
        for i in range(relevant_tokens_num):
            relevant_word_log_probs.append(generate_logprobs[i, relevant_tokens[i]].item())
        print("relevant word log probs ", relevant_word_log_probs)
        # print("indices of nonzero generation scores ", (logits > -torch.inf).nonzero())        
        print("input_ids_continuation[0][-1] ", input_ids_continuation)
        print("outputs generate scores shape ", len(outputs_generate.scores), outputs_generate.scores[0][0].shape, outputs_generate.scores[0].shape)
        print("Answer logit retrieved with og Jenn's method ", relevant_word_log_probs)
        first_log_probs_from_logits = sum(relevant_word_log_probs)
        first_mean_log_probs_from_logits = np.mean(relevant_word_log_probs)
        print("Logits transformed to log probs ", first_log_probs_from_logits, first_mean_log_probs_from_logits)
    else:
        generated_continuation = "NONE"
        first_log_probs_from_logits = None
    ############# END of production-task specific exploration ########
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
    # print("Relevant labels ", relevant_labels)
    output_masked = model(input_ids, labels=relevant_labels)
    print("Output loss (i.e., mean) computed with NPNLG approach ", output_masked.loss.item(), output_masked.loss.item() * (input_ids_continuation.shape[-1]))
    # for doubke checking, compute the same with only the last tokens
    print("input ids continuation for checking NPNLG code" , input_ids_continuation)
    # output_last_tokens_loss = model(input_ids_continuation.unsqueeze(0), labels=input_ids_continuation.unsqueeze(0))
    # print("npnlg double checking loss ", output_last_tokens_loss.loss.item())

    return meanLogProb, output_masked.loss.item(), generated_continuation, first_log_probs_from_logits
            

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
        alpha_interpretation,
        computation,
    ):

    # take care of special tokens for chat models
    # assume that the task and context come from the user, and the response from the model
    # no specific system prompt is passed
    # if one wanted to, the expected formatting would be: [INST]<<SYS>>{system prompt}<</SYS>>\n\n{user message}[/INST]
    if "chat" in model_name:
        context_production = f"[INST]{vignette['context_production']}[/INST]"
        context_interpretation = f"[INST]{vignette['context_interpretation']}[/INST]"
    else:
        context_production = vignette['context_production'] #.replace("I would choose the word ", "")
        context_interpretation = vignette['context_interpretation'] #.replace("My friend wants to refer to ", "")
    # general sanity check
    #testing_prompt = getLogProbContinuation(
    #    "Continue the following Christmas song: Dashing through the snow\nIn a one-horse open sleigh\nO'er the fields we go\nLaughing all the way\nBells on bobtails ring\nMaking spirits bright\nWhat fun it is to ride and sing\nA sleighing song tonight\nJingle bells, Jingle bells\n", "Jingle",
    #    model, tokenizer)
    #print("Jingle continuation " ,testing_prompt) 
    #testing_prompt2 = getLogProbContinuation(
    #    "Continue the following Christmas song: Dashing through the snow\nIn a one-horse open sleigh\nO'er the fields we go\nLaughing all the way\nBells on bobtails ring\nMaking spirits bright\nWhat fun it is to ride and sing\nA sleighing song tonight\nJingle bells, Jingle bells\n", "Christmas",
    #    model, tokenizer)
    #print("Christmas continuation ", testing_prompt2)
    if computation == "use_own_scoring":
        # production
        prod_lprob_target, prod_lprob_target_gen, gen_seq_target, gen_p_target      = getLogProbContinuation(
            context_production,'"' + vignette["production_target"] + '"',
            model, tokenizer)
        prod_lprob_competitor, prod_lprob_competitor_gen, gen_seq_competitor, gen_p_competitor   = getLogProbContinuation(
            context_production,'"' + vignette["production_competitor"]+ '"',
            model, tokenizer)
        prod_lprob_distractor1, prod_lprob_distractor1_gen, gen_seq_distractor1, gen_p_distractor1  = getLogProbContinuation(
            context_production, '"' + vignette["production_distractor1"] + '"',
            model, tokenizer)
        prod_lprob_distractor2, prod_lprob_distractor2_gen, gen_seq_distractor2, gen_p_distractor2  = getLogProbContinuation(
            context_production, '"' + vignette["production_distractor2"] + '"',
            model, tokenizer)
        # for testing, also just sample a few productions
        predictions_prompt_ids = tokenizer(context_production.strip(), return_tensors="pt").to("cuda:0")
        production_samples = model.generate(
            **predictions_prompt_ids,
            max_new_tokens=16,
            do_sample = False,
            #    temperature = 0.7,
        )
        production_decoded = tokenizer.batch_decode(production_samples)
        print("### productions_decoded", production_decoded)
        
        # interpretation

        int_lprob_target, int_lprob_target_gen, gen_seq_target_int, gen_p_target_int       = getLogProbContinuation(
            context_interpretation,'"' +  vignette["interpretation_target"] + '"',
            model, tokenizer)
        int_lprob_competitor, int_lprob_comp_gen, gen_seq_competitor_int, gen_p_competitor_int   = getLogProbContinuation(
            context_interpretation, '"' +vignette["interpretation_competitor"] + '"',
            model, tokenizer)
        int_lprob_distractor, int_lprob_distractor_gen, gen_seq_distractor_int, gen_p_distractor_int   = getLogProbContinuation(
            context_interpretation,'"' + vignette["interpretation_distractor"] + '"',
            model, tokenizer)
        predictions_interpretation_ids = tokenizer(context_interpretation.strip(), return_tensors="pt").to("cuda:0")
        interpretation_samples = model.generate(
            **predictions_interpretation_ids,
            max_new_tokens=16,
            do_sample = False,
            #    temperature = 0.7,
        )
        interpretation_decoded = tokenizer.batch_decode(interpretation_samples)
        print("### Interpretation_decoded ", interpretation_decoded)

    elif computation == "use_surprisal":
        prod_lprob_target, prod_lprob_target_gen      = use_surprisal(
            context_production,'"' + vignette["production_target"] + '"',
            model, tokenizer)
        prod_lprob_competitor, prod_lprob_competitor_gen  = use_surprisal(
            context_production,'"' + vignette["production_competitor"]+ '"',
            model, tokenizer)
        prod_lprob_distractor1, prod_lprob_distractor1_gen = use_surprisal(
            context_production, '"' + vignette["production_distractor1"] + '"',
            model, tokenizer)
        prod_lprob_distractor2, prod_lprob_distractor2_gen = use_surprisal(
            context_production, '"' + vignette["production_distractor2"] + '"',
            model, tokenizer)
        production_decoded = ""
        int_lprob_target, int_lprob_target_gen      = use_surprisal(
            context_interpretation,'"' +  vignette["interpretation_target"] + '"',
            model, tokenizer)
        int_lprob_competitor, int_lprob_comp_gen  = use_surprisal(
            context_interpretation, '"' +vignette["interpretation_competitor"] + '"',
            model, tokenizer)
        int_lprob_distractor, int_lprob_distractor_gen  = use_surprisal(
            context_interpretation,'"' + vignette["interpretation_distractor"] + '"',
            model, tokenizer)
        interpretation_decoded = ""

    elif computation == "use_jenns_method":
        prod_lprob_target, prod_lprob_target_gen      = use_jenns_method(
            context_production,'"' + vignette["production_target"] + '"',
            model, tokenizer)
        prod_lprob_competitor, prod_lprob_competitor_gen  = use_jenns_method(
            context_production,'"' + vignette["production_competitor"]+ '"',
            model, tokenizer)
        prod_lprob_distractor1, prod_lprob_distractor1_gen = use_jenns_method(
            context_production, '"' + vignette["production_distractor1"] + '"',
            model, tokenizer)
        prod_lprob_distractor2, prod_lprob_distractor2_gen = use_jenns_method(
            context_production, '"' + vignette["production_distractor2"] + '"',
            model, tokenizer)
        production_decoded = ""
        int_lprob_target, int_lprob_target_gen      = use_jenns_method(
            context_interpretation,'"' +  vignette["interpretation_target"] + '"',
            model, tokenizer)
        int_lprob_competitor, int_lprob_comp_gen  = use_jenns_method(
            context_interpretation, '"' +vignette["interpretation_competitor"] + '"',
            model, tokenizer)
        int_lprob_distractor, int_lprob_distractor_gen  = use_jenns_method(
            context_interpretation,'"' + vignette["interpretation_distractor"] + '"',
            model, tokenizer)
        interpretation_decoded = ""

    else:
        raise ValueError("Computation method not recognized. Please use 'use_own_scoring' or 'use_surprisal'.")

    scores_production = np.array([prod_lprob_target, prod_lprob_competitor, prod_lprob_distractor1, prod_lprob_distractor2])
    probs_production = soft_max(scores_production, alpha=alpha_production)
    # softmax the scores generated with alternative method
    scores_production_gen = np.array([prod_lprob_target_gen, prod_lprob_competitor_gen, prod_lprob_distractor1_gen, prod_lprob_distractor2_gen])
    probs_production_gen = soft_max(scores_production_gen, alpha=alpha_interpretation)
    scores_interpretation = np.array([int_lprob_target, int_lprob_competitor, int_lprob_distractor])
    probs_interpretation = soft_max(scores_interpretation, alpha=1)

    scores_interpretation_gen = np.array([int_lprob_target_gen, int_lprob_comp_gen, int_lprob_distractor_gen])
    probs_interpretation_gen = soft_max(scores_interpretation_gen, alpha=alpha_interpretation)

    output_dict = {
        'alpha_production'              : alpha_production,
        'scores_production_target'      : scores_production[0],
        'scores_production_competitor'  : scores_production[1],
        'scores_production_distractor1' : scores_production[2],
        'scores_production_distractor2' : scores_production[3],

        'scores_production_target_npnlg'      : gen_p_target, #scores_production_gen[0],
        'scores_production_competitor_npnlg'  : gen_p_competitor, #scores_production_gen[1],
        'scores_production_distractor1_npnlg' : gen_p_distractor1, #scores_production_gen[2],
        'scores_production_distractor2_npnlg' : gen_p_distractor2,#scores_production_gen[3],

        'prob_production_target'        : probs_production[0],
        'prob_production_competitor'    : probs_production[1],
        'prob_production_distractor1'   : probs_production[2],
        'prob_production_distractor2'   : probs_production[3],
        'prob_production_target_npnlg'        : gen_seq_target, #probs_production_gen[0],
        'prob_production_competitor_npnlg'    : gen_seq_competitor, #probs_production_gen[1],
        'prob_production_distractor1_npnlg'   : gen_seq_distractor1, #probs_production_gen[2],
        'prob_production_distractor2_npnlg'   : gen_seq_distractor2, #probs_production_gen[3],
        'production_decoded': "\n".join(production_decoded),
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
        'prob_interpretation_distractor_npnlg'   : probs_interpretation_gen[2],
        'interpretation_decoded': "\n".join(interpretation_decoded),
    }

    return output_dict

def use_surprisal(
        initialSequence, 
        continuation, 
        model,
        tokenizer,
        preface = ''):
    """
    Helper for retrieving log probability with package surprisal
    https://github.com/aalok-sathe/surprisal
    """

    initialSequence = preface + initialSequence
    prompt = preface + initialSequence + continuation
    print(initialSequence)
    print(prompt)
    # get surprisal of entire sequence
    [surpr] = model.surprise(prompt)
    print("Surprisal computed with surprisal package ", surpr)
    # tokenizee schole prompt and input only separately to be able to pull last continuation
    # tokens only
    input_ids_prompt = tokenizer(
        initialSequence.strip(),
        return_tensors="pt",
    ).input_ids
    input_ids = tokenizer(
        prompt.strip(),
        return_tensors="pt",
    ).input_ids
    input_chars = len(initialSequence)
    prompt_chars = len(prompt)
    continuation_num_chars = prompt_chars - input_chars
    print("continuation tokens pulled for computation with surprisal package ", input_chars, prompt_chars, continuation_num_chars)
    sum_surprisal = surpr[input_chars:prompt_chars] 
    print("Sum surprisals ", sum_surprisal)
    meanLogP = - sum_surprisal / continuation_num_chars
    print("Mean log prob ", meanLogP)
    return meanLogP, -sum_surprisal

def use_jenns_method(
        initialSequence, 
        continuation, 
        model,
        tokenizer,
        preface = ''
    ):
    """
    Helper for retrieving log probability with Jennifer Hu's method
    from this paper: https://github.com/aalok-sathe/surprisal
    """
    initialSequence = preface + initialSequence
    prompt = preface + initialSequence + continuation
    
    input_ids_prompt = tokenizer(initialSequence.strip(), return_tensors="pt").input_ids.to(model.device)
    input_ids = tokenizer(prompt.strip(), return_tensors="pt").input_ids.to(model.device)
    labels = tokenizer(prompt.strip(), return_tensors="pt").input_ids.to(model.device)
    mask = []
    print(input_ids_prompt)
    print(input_ids)
    # get the tokens of the initial sequence that we do not want to retrieve log probs for
    # NB: we cannot mask based on particular token ID because the same tokens might be re-used in the prompt
    # therefore, the masking is based on index of the tokens
    # Tokenize the inputs and labels.
    max_id_prompt = input_ids_prompt.shape[-1] - 1
    print("max_id_prompt ", max_id_prompt)
    for i, _ in enumerate(input_ids[0, 1:]):
        mask.append(i >= max_id_prompt)
    mask_tensor = torch.BoolTensor(mask).to(model.device)
    print("Mask tensor ",mask_tensor.shape,  mask_tensor)
    # Model forward.
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels)

    # Turn logits into log probabilities.
    logits = outputs.logits
    logprobs = torch.nn.functional.log_softmax(logits, dim=-1)[:, :-1]
    print('logprobs shape ', logprobs.shape)
    # Subset the labels and logprobs we care about,
    # i.e. the non-"special" tokens (e.g., "<extra_id_0>").
    # mask = torch.BoolTensor([tok_id not in self.ids_to_ignore for tok_id in labels[0]])
    # TODO: note that the logprobs are not shifted!
    relevant_labels = labels[0, 1:][mask_tensor]
    relevant_logprobs = logprobs[0][mask_tensor].cpu()
    print("relevant labels ", relevant_labels)
    print("relevant logprobs ", relevant_logprobs)
    # Index into logprob tensor using the relevant token IDs.
    logprobs_to_sum = [
        relevant_logprobs[i][tok_id] 
        for i, tok_id in enumerate(relevant_labels)
    ]
    total_logprob = sum(logprobs_to_sum).item()
    print("toal log prob ", total_logprob)
    avg_logprob = np.mean(logprobs_to_sum).item()
    print("avg log prob ", avg_logprob)

    return total_logprob, avg_logprob

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
    elif computation == "use_surprisal":
        model = AutoHuggingFaceModel.from_pretrained(model_name, model_class="gpt", precision='fp16')
        model.to('cuda')
        [s] = model.surprise("The cat is on the mat")
        print("Cat check ", [s])
        print("Cat word aggregation ", s[3:6, "word"])
        print("Cat char agregation ", s[3:6] )
    else:
        raise ValueError("Computation method not recognized. Please use 'use_own_scoring' or 'use_surprisal'.")

    list_of_dicts = []

    # for comparability of results, use materials from GPT-3 results
    if task == 'ref_game':
        vignettes = pd.read_csv('../02-data/results_GPT.csv')
        # vignettes = pd.read_csv('../02-data/sanity_check_data.csv')
        for i, vignette in tqdm(vignettes.iterrows()):
            predictions = get_model_predictions(
                vignette, 
                model, 
                tokenizer, 
                model_name,
                0.5, 
                0.5,
                computation,
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

    #    pprint(results_df)
        # continuous saving of results
            results_name = f'results_oldJennsMethodWithGenerate_wQuots_{computation}_{name_for_saving}_{date_out}.csv'
            results_df.to_csv(results_name, index = False)
    elif task == "sanity_check":
        vignettes = pd.read_csv('../02-data/sanity_check_data.csv')
        for i, vignette in tqdm(vignettes.iterrows()):
        #    predictions = get_model_predictions(
        #        vignette, 
        #        model, 
        #        tokenizer, 
        #        model_name,
        #        0.5, 
        #        0.5
        #    )

            materials = {
                'trial': i,	
                'interpretation_target': vignette['interpretation_target'],	
                'interpretation_competitor': vignette['interpretation_competitor'],	
                'interpretation_distractor': vignette['interpretation_distractor'],	
                # 'interpretation_index_target': vignette['interpretation_index_target'],	
                # 'interpretation_index_competitor': vignette['interpretation_index_competitor'],	
                # 'interpretation_index_distractor': vignette['interpretation_index_distractor'],	
                'context_interpretation': vignette['context_interpretation'],
                'scores_interpretation_target': getLogProbContinuation(
                    vignette['context_interpretation'],  vignette["interpretation_target"] ,
                    model, tokenizer)[0],
                'scores_interpretation_competitor': getLogProbContinuation(
                    vignette['context_interpretation'],  vignette["interpretation_competitor"] ,
                    model, tokenizer)[0],
                'scores_interpretation_distractor': getLogProbContinuation(
                    vignette['context_interpretation'],  vignette["interpretation_distractor"] ,
                    model, tokenizer)[0],
                
            }
            output = dict(**materials) #, **predictions)
            list_of_dicts.append(output)

            results_df = pd.DataFrame(list_of_dicts)

    #    pprint(results_df)
        # continuous saving of results
            results_name = f'results_sanity_check_{name_for_saving}_{date_out}.csv'
            results_df.to_csv(results_name, index = False)
    else:
        raise ValueError("Task not recognized. Please use 'ref_game' or 'sanity_check'.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="meta-llama/Llama-2-7b-hf", 
        help="Model name"
    )
    parser.add_argument(
        "--task", 
        type=str, 
        default="ref_game", 
        help="Task to run. Either 'ref_game' or 'sanity_check'."
    )
    parser.add_argument(
        "--computation", 
        type=str, 
        default="use_own_scoring", 
        help="Type of score retrieval implementation to use."
    )

    args = parser.parse_args()

    main(
        args.model_name, 
        args.task,
        args.computation,
    )
