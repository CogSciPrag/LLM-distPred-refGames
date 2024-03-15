# Different retrieval and aggregation methods

The prompt with quotation marks was chosen as the "final" formatting and different programmatic ways of retrieving the different answer options in quotes were explored. 

The single-token log probabilities were, additionally, aggregated in two different ways to results in option-level scores: via (1) summing ("sum_" directories contain respective results) or via (2) averaging over the tokens ("mean_" directories contain respective results). The main results intended for the paper are under **mean_scores_forward**.

Further, the directory names reflect the method of retrieving the scores.

* `mean_scores_forward`: the score for each answer option was retrieved with code written by PT. The following steps are executed:
  * the entire prompt (ending in the option to be scored) is used for a forward pass through the model (under no grad decorator)
  * the logits are retrieved from the output and log_softmax-ed.
  * the scores are shifted relative to the tokens (i.e., logit tensor at index 0 contains scores for tokens at index 1 given token and index 0)
  * the logits corresponding to the indices of the option tokens with the appropriate shift are used to read out the scores of the respective tokens
  * the scores for all tokens belonging to the answer option are averaged. (*NB*: the averages were derived from the original summed scores)
* `sum_scores_forward`: the score for each answer option was retrieved with code written by PT. The steps are identical to the ones described above, except for the last step: the single token log probabilities are summed to retrieve the option log probability.
* `mean_scores_masked`: the score for each answer option was retrieved with code based on examples from the NPNLG class. The steps are the following:
  * the entire prompt (including the option) is passed to the forward method of the model along with masked labels. Specifically, all tokens except for the option tokens are masked with -100.
  * the loss is retrieved from this forward pass.
  * the loss corresponds to the average negative log probability of the non-masked tokens.
* `mean_scores_generate`: the score was retrieved based on the values returned under "scores" when the HuiggingFace `.generate()` utility is used. This is based on the method used for huggingface models by Jenn. **NOTE:** this was only explored for a few base models and for the production task only. Some cells contain None values of the scores which indicates that the prediction deviated from <"option"> format. The file formatting is intended for visual inspection and also *slightly differs* (ask Polina).
  * here, the prompt *without* the option to be scored was passed as input to .generate().
  * maximally five tokens were generated and the scores of the generated sequence were retrieved.
  * here, no shifting of the logits was applied since only the scores of the completion tokens are available. 
  * scores for the generated continuation of the form "option" were retreived. The purpose of the exploration was to see if the scores of the same option under free generation and the retrieval are identical. Results were different after the ~4th decimal.  
  * the tokens belonging to one option were averaged.
* `sum_scores_surprisal`: for exploration, the scores were computed with the [surprisal package](https://github.com/aalok-sathe/surprisal) which implements retrieval of surprisals (i.e., negative log probability) of the tokens (at least for GPT-2 and BERT). However, since the package only allows to reliably specify the *characters* for which the log probabilities should be summed and returned (not particular words or tokens), there is no explicit Llama support and the resulting numbers here are very different from all methods above, they are *not to be taken seriously*.
