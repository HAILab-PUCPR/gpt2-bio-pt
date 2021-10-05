# -*- coding: utf-8 -*-

from transformers import AutoTokenizer

# vamos usar o mesmo tokenizador do gpt2-small-portuguese
tokenizer = AutoTokenizer.from_pretrained("pierreguillou/gpt2-small-portuguese")

# para gerar estatisticas do treinamento, mas precisa ter cadastro no wandb
# pode comentar essas linhas caso não tenha cadastro
import wandb

wandb.init(project='gpt2-bio-pt', entity='gpt2')

import re
from sklearn.model_selection import train_test_split

# abrindo o dataset com os textos, no nosso caso, biomedicos
file1 = open("dataset.txt", encoding='utf-8')
FileContent = file1.read()

# quebrando frases pelo ponto
frases = FileContent.split(". ")

# arquivos de treinamento e teste
train, test = train_test_split(frases,test_size=0.15) 

print("Train dataset length: "+str(len(train)))
print("Test dataset length: "+ str(len(test)))

def build_text_files(data_txt, dest_path):
    f = open(dest_path, 'w')
    data = ''
    # limpeza dos dados
    for texts in data_txt:
        summary = texts.strip()
        summary = re.sub(r"\s", " ", summary)
        data += summary + "  "
    f.write(data)


build_text_files(test,'test_dataset.txt')
build_text_files(train,'train_dataset.txt')

train_path = 'train_dataset.txt'
test_path = 'test_dataset.txt'

from transformers import TextDataset,DataCollatorForLanguageModeling

def load_dataset(train_path,test_path,tokenizer):
    train_dataset = TextDataset(
          tokenizer=tokenizer,
          file_path=train_path,
          block_size=128)
     
    test_dataset = TextDataset(
          tokenizer=tokenizer,
          file_path=test_path,
          block_size=128)   
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )
    return train_dataset,test_dataset,data_collator

train_dataset,test_dataset,data_collator = load_dataset(train_path,test_path,tokenizer)

"""# Initialize `Trainer` with `TrainingArguments` and GPT-2 model

The [Trainer](https://huggingface.co/transformers/main_classes/trainer.html#transformers.Trainer) class provides an API for feature-complete training. It is used in most of the [example s](https://huggingface.co/transformers/examples.html) from Huggingface. Before we can instantiate our `Trainer` we need to download our GPT-2 model and create a [TrainingArguments](https://huggingface.co/transformers/main_classes/trainer.html#transformers.TrainingArguments) to access all the points of customization during training. In the `TrainingArguments`, we can define the Hyperparameters we are going to use in the training process like our `learning_rate`, `num_train_epochs`, or  `per_device_train_batch_size`. A complete list can you find [here](https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments).
"""

from transformers import Trainer, TrainingArguments,AutoModelWithLMHead

# vamos fazer fine tuning a partir do gpt2-small-portuguese
model = AutoModelWithLMHead.from_pretrained("pierreguillou/gpt2-small-portuguese")

# parametros do modelo
training_args = TrainingArguments(
    output_dir="./modelo", #The output directory
    overwrite_output_dir=True, #overwrite the content of the output directory
    num_train_epochs=3, # number of training epochs
    per_device_train_batch_size=32, # batch size for training
    per_device_eval_batch_size=64,  # batch size for evaluation
    eval_steps = 400, # Number of update steps between two evaluations.
    save_steps=800, # after # steps model is saved 
    warmup_steps=500,# number of warmup steps for learning rate scheduler
    run_name = 'gpt2',   # name of the W&B run (optional) -> comente aqui se não for usar o wandb
    do_train = True,
    do_eval = True,
    #evaluation_during_training = True
    )


trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    #return_outputs=True
)

"""# Train and save the model

To train the model we can simply run `Trainer.train()`.
"""

# treinando o modelo
train_result = trainer.train()

# salvando o modelo
trainer.save_model()

print('-------train_result---------')
print(train_result)

import math
eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

print('-------eval_result---------')
print(eval_results)

if training_args.do_train:
    print('metrics traning')
    metrics = train_result.metrics

    trainer.log_metrics("train", metrics)
    print('salvando metricas train')
    trainer.save_metrics("train", metrics)
    trainer.save_state()

# Evaluation
if training_args.do_eval:
    print('evaluate')
    metrics = trainer.evaluate()

    #max_val_samples = training_args.max_val_samples if training_args.max_val_samples is not None else len(eval_dataset)
    #metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))
    #perplexity = math.exp(metrics["eval_loss"])
    #metrics["perplexity"] = perplexity

    trainer.log_metrics("eval", metrics)
    print('salvando metricas eval')

    trainer.save_metrics("eval", metrics)
