# Refgame 

This directory contains results for the main reference game that vary in (1) aggregation method of single tokens for one option and (2) in methodology of how the log probabilities were (programatically) retrieved from the different Llama models.
Furthermore, different formatting of the prompt was explored (see below); for both prompt types, depending on the model special `[INST]` tokens were inserted.

`long_input`; Here, not just the answer option, but the entire sentence 'I would choose the word <option>' was scored.

`with_quotes`: The scored answer options were presented in quotation marks like so: 
> Your task is to play a conversation game. There are three objects that you and your friend can see. You have to choose a single word to identify one of the three objects for your friend.
>
> The three objects are:
>
> a green triangle with dots
> a green circle with dots
> a red triangle with dots
>
> Your task is to make your friend pick out the following target object:
>
> the red triangle with dots
>
> Which of the following words would you choose:
>
> green
> triangle
> red
> circle
>
> Your answer:
>
> I would choose the word **"red"**
`without_quotes`: The scored answer options were presented in quotation marks like so: 
> Your task is to play a conversation game. There are three objects that you and your friend can see. You have to choose a single word to identify one of the three objects for your friend.
>
> The three objects are:
>
> a green triangle with dots
> a green circle with dots
> a red triangle with dots
>
> Your task is to make your friend pick out the following target object:
>
> the red triangle with dots
>
> Which of the following words would you choose:
>
> green
> triangle
> red
> circle
>
> Your answer:
>
> I would choose the word **red**