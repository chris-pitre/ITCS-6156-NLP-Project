from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, get_linear_schedule_with_warmup
from dataset import DialogueDataset
from torch.utils.data import random_split, DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW
import pandas as pd
import torch
import numpy as np
import random
import os

pwd = os.path.dirname(__file__)
models_path = os.path.join(pwd, './saved_models/')

device = torch.device("cuda")

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

def generate_response(prompt: str) -> str:
    pwd = os.path.dirname(__file__)
    models_path = os.path.join(pwd, './saved_models/example_generative_training_weights')
    tokenizer = GPT2Tokenizer.from_pretrained(models_path)
    configuration = GPT2Config.from_pretrained(models_path, output_hidden_states=False)
    model = GPT2LMHeadModel.from_pretrained(models_path, config=configuration)

    model.resize_token_embeddings(len(tokenizer))

    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Generate text based on the prompt
    output = model.generate(input_ids,
                            do_sample=True,
                            max_length=100,
                            pad_token_id=tokenizer.eos_token_id
                            )


    # Decode and print the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text



def train_model():
    """
    Training script made with help from this notebook: https://colab.research.google.com/drive/13dZVYEOMhXhkXWfvSMVM1TTtUDrT6Aeh?usp=sharing#scrollTo=6ulTWaOr8QNY
    """
    # Load pretrained GPT-2 model and tokenizer
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name, bos_token='<bos>', eos_token='<eos>', pad_token='<pad>')
    configuration = GPT2Config.from_pretrained(model_name, output_hidden_states=False)
    model = GPT2LMHeadModel.from_pretrained(model_name, config=configuration)

    model.resize_token_embeddings(len(tokenizer))

    data = pd.read_csv("src/Python-NLP/datasets/dialogue.csv").to_numpy()
    dataset = DialogueDataset(data, tokenizer)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(
        train_dataset,
        sampler = RandomSampler(train_dataset),
        batch_size = 2
    )

    val_dataloader = DataLoader(
        val_dataset,
        sampler = SequentialSampler(val_dataset),
        batch_size = 2
    )

    # Hyperparameters
    epochs = 5
    lr = 5e-4
    warmup_steps = 1e2
    epsilon = 1e-8
    total_steps = len(train_dataloader) * epochs

    optimizer = AdamW(model.parameters(), lr=lr, eps=epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    print("Information Finetuning")

    for epoch in range(epochs):
        print(f"Epoch: {epoch+1}")
        
        total_train_loss = 0
        model = model.to(device)
        model.train()
        for step, batch in enumerate(train_dataloader):
            b_input_ids = batch[0].to(device)
            b_labels = batch[0].to(device)
            b_masks = batch[1].to(device)

            model.zero_grad()

            outputs = model(b_input_ids, labels=b_labels, attention_mask=b_masks, token_type_ids=None)
            loss = outputs[0]
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_dataloader)

        print(f"Training Loss: {avg_train_loss}")

        total_val_loss = 0
        model.eval()
        for step, batch in enumerate(val_dataloader):            
            b_input_ids = batch[0].to(device)
            b_labels = batch[0].to(device)
            b_masks = batch[1].to(device)
            with torch.no_grad():
                outputs = model(b_input_ids, labels=b_labels, attention_mask=b_masks, token_type_ids=None)
                loss = outputs[0]
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_dataloader)
        print(f"Validation Loss: {avg_val_loss}")
        model.cpu()

    if not os.path.exists(models_path):
        os.makedirs(models_path)

    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(models_path)
    tokenizer.save_pretrained(models_path)

# run file by itself to train weights
if __name__ == "__main__":
    train_model()

    print(generate_response("Question: Where is the key? Answer: "))