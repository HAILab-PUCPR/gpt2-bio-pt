<img src="./img/logo-gpt2-bio-pt.png" alt="Logo GPT2-Bio-Pt">

# GPT2-Bio-Pt - a Language Model for Portuguese Biomedical text generation

## Introduction

GPT2-Bio-Pt (Portuguese Biomedical GPT-2 small) is a language model for Portuguese based on the OpenAI GPT-2 model, trained from the [GPorTuguese-2](https://huggingface.co/pierreguillou/gpt2-small-portuguese/) with biomedical literature.

We used **Transfer Learning and Fine-tuning techniques** with 110MB of training data, corresponding to 16,209,373 tokens and 729,654 sentences. 

## GPT-2 

*Note: information copied/pasted from [Model: gpt2 >> GPT-2](https://huggingface.co/gpt2#gpt-2)*

Pretrained model on English language using a causal language modeling (CLM) objective. It was introduced in this [paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) and first released at this [page](https://openai.com/blog/better-language-models/) (February 14, 2019).

<img src="./img/openAi.png" alt="OpenAI GPT-2">
The OpenAI GPT pre-training architecture, which trains left-to-right Transformer decoders on a large corpus.

## Model description

*Note: information copied/pasted from [Model: gpt2 >> Model description](https://huggingface.co/gpt2#model-description)*

GPT-2 is a transformers model pretrained on a very large corpus of English data in a self-supervised fashion. This means it was pretrained on the raw texts only, with no humans labelling them in any way (which is why it can use lots of publicly available data) with an automatic process to generate inputs and labels from those texts. More precisely, it was trained to guess the next word in sentences.

More precisely, inputs are sequences of continuous text of a certain length and the targets are the same sequence, shifted one token (word or piece of word) to the right. The model uses internally a mask-mechanism to make sure the predictions for the token `i` only uses the inputs from `1` to `i` but not the future tokens.

This way, the model learns an inner representation of the English language that can then be used to extract features useful for downstream tasks. The model is best at what it was pretrained for however, which is generating texts from a prompt.

<img src="./img/comparacaoGPT2.png" alt="GPT2-Bio-Pt x Portuguese GPT-2" width="80%">
Synthetic text samples generated by our model (biomedical domain) and GPT2-Portuguese model (generic domain).


## How to use GPT2-Bio-Pt with HuggingFace

```
from transformers import pipeline

chef = pipeline('text-generation',model=r"pucpr/gpt2-bio-pt", tokenizer=r"pucpr/gpt2-bio-pt",config={'max_length':800})

result = chef('O paciente chegou no hospital')[0]['generated_text']
print(result)

```

Resultado:

*```O paciente chegou no hospital três meses após a operação, não houve complicações graves.  Entre os grupos que apresentaram maior número de lesões, o exame da cavidade pélvica estava significantemente associado à ausência de complicações.  Foi encontrada uma maior incidência de fraturas (...)```*


## Citation
```
@INPROCEEDINGS{9474713,
  author={Schneider, Elisa Terumi Rubel and de Souza, João Vitor Andrioli and Gumiel, Yohan Bonescki and Moro, Claudia and Paraiso, Emerson Cabrera},
  booktitle={2021 IEEE 34th International Symposium on Computer-Based Medical Systems (CBMS)}, 
  title={A GPT-2 Language Model for Biomedical Texts in Portuguese}, 
  year={2021},
  volume={},
  number={},
  pages={474-479},
  doi={10.1109/CBMS52027.2021.00056}}
```
