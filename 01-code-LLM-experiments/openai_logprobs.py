import openai
import time
import numpy as np
import pandas as pd
import os
import argparse
from tqdm import tqdm
import openai
from pprint import pprint
# import make_material
from make_material import sample_vignette

# openai.api_key = os.getenv("OPENAI_API_KEY")
# OAI = "bla"
openai.api_key = OAI

def getLogProbContinuation(initialSequence, continuation, preface = ''):
    """
    Helper for retrieving log probability of different response types from GPT-3.
    """
    initialSequence = preface + initialSequence
    response = openai.Completion.create(
            engine      = "text-davinci-003",
            prompt      = initialSequence + " " + continuation,
            max_tokens  = 0,
            temperature = 1,
            logprobs    = 0,
            echo        = True
        )
    text_offsets = response.choices[0]['logprobs']['text_offset']
    cutIndex = text_offsets.index(max(i for i in text_offsets if i < len(initialSequence))) + 1
    endIndex = response.usage.total_tokens
    answerTokens = response.choices[0]["logprobs"]["tokens"][cutIndex:endIndex]
    answerTokenLogProbs = response.choices[0]["logprobs"]["token_logprobs"][cutIndex:endIndex]
    meanAnswerLogProb = np.mean(answerTokenLogProbs)
    sentenceLogProb = np.sum(answerTokenLogProbs)

    return meanAnswerLogProb, sentenceLogProb, (endIndex - cutIndex)

def soft_max(scores, alpha=1):
    scores = np.array(scores)
    output = np.exp(scores * alpha)
    return(output / np.sum(output))


def get_model_predictions(vignette, alpha_production, alpha_interpretation):

    # production

    lprob_target      = getLogProbContinuation(vignette['context_production'], vignette["production_target"])
    lprob_competitor  = getLogProbContinuation(vignette['context_production'], vignette["production_competitor"])
    lprob_distractor1 = getLogProbContinuation(vignette['context_production'], vignette["production_distractor1"])
    lprob_distractor2 = getLogProbContinuation(vignette['context_production'], vignette["production_distractor2"])

    scores_production = np.array([lprob_target, lprob_competitor, lprob_distractor1, lprob_distractor2])[:,1]
    probs_production = soft_max(scores_production, alpha_production)

    # interpretation

    lprob_target      = getLogProbContinuation(vignette['context_interpretation'], vignette["interpretation_target"])
    lprob_competitor  = getLogProbContinuation(vignette['context_interpretation'], vignette["interpretation_competitor"])
    lprob_distractor  = getLogProbContinuation(vignette['context_interpretation'], vignette["interpretation_distractor"])

    scores_interpretation = np.array([lprob_target, lprob_competitor, lprob_distractor])[:,1]
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

    return(output_dict)


n_trials = 100

list_of_dicts = []

for i in tqdm(range(n_trials)):
    vignette = sample_vignette()
    predictions = get_model_predictions(vignette, 0.5, 0.5)
    trial = {'trial': i}
    output = dict(**trial, **vignette, **predictions)
    list_of_dicts.append(output)

results_df = pd.DataFrame(list_of_dicts)

pprint(results_df)

results_df.to_csv('results.csv', index = False)
