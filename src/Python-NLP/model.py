from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, get_linear_schedule_with_warmup
from dataset import DialogueDataset, ClassificationDataset
from torch.utils.data import random_split, DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd
import torch
import numpy as np
import random
import os
import re
import matplotlib.pyplot as plt

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

def classify_response(prompt: str) -> str:
    pwd = os.path.dirname(__file__)
    models_path = os.path.join(pwd, './saved_models/example_classification_training_weights')
    tokenizer = GPT2Tokenizer.from_pretrained(models_path)
    configuration = GPT2Config.from_pretrained(models_path, output_hidden_states=False)
    model = GPT2LMHeadModel.from_pretrained(models_path, config=configuration)

    model.resize_token_embeddings(len(tokenizer))
    return classify_response_no_load(prompt, tokenizer, model)

def class_response(label: str) -> str:
    small_talk = [
                    "You are asking the wrong questions", 
                    "What a meaningless question", 
                    "The winds of fate are turbelent. There are times when even I cannot see what lies ahead",
                    "What a great question.\n\n It has an answer, I'm sure..."
    ]
    match label:
        case "key": return "You will find the key hidden behind the tree"
        case "door": return "The door may be locked, but all locks have a key"
        case "location": return "These are the ruins of Eldoria"
        case "identity": return "I am the Oracle of Eldoria"
        case "goal": return "To progress on your journey, you will need to travel north"
        case "small_talk": return random.choice(small_talk)
        case _ : return "Now I have become error"


    
def classify_response_no_load(prompt: str, tokenizer: GPT2Tokenizer, model: GPT2LMHeadModel) -> str:
    input_ids = tokenizer.encode(f"<bos>Question: {prompt}\n", return_tensors="pt")

    # Generate text based on the prompt
    output = model.generate(input_ids,
                            do_sample=True,
                            max_length=75,
                            top_k=50,
                            top_p=0.90,
                            temperature=0.1,
                            num_return_sequences=1,
                            pad_token_id=tokenizer.eos_token_id
                            )

    # Decode and print the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    try:
        text_class = re.findall("\nClass: (.*)", generated_text)[-1]
    except:
        text_class = "small_talk"
    return text_class


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

def train_model_classification():
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name, bos_token='<bos>', eos_token='<eos>', pad_token='<pad>')
    configuration = GPT2Config.from_pretrained(model_name, output_hidden_states=False)
    model = GPT2LMHeadModel.from_pretrained(model_name, config=configuration)

    model.resize_token_embeddings(len(tokenizer))

    pwd = os.path.dirname(__file__)
    data_path = os.path.join(pwd, './datasets/classification_dialogue.csv')
    data = pd.read_csv(data_path).to_numpy()
    dataset = ClassificationDataset(data, tokenizer)
    train_size = int(0.8 * len(dataset))
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

    total_train_loss_list = []
    total_train_acc = []
    total_train_recall = []
    total_train_precision = []
    total_train_f1 = []

    total_val_loss_list = []
    total_val_acc = []
    total_val_recall = []
    total_val_precision = []
    total_val_f1 = []

    for epoch in range(epochs):
        print(f"Epoch: {epoch+1}")
        
        total_train_loss = 0
        all_preds = []
        all_original = []

        model = model.to(device)
        model.train()
        for step, batch in enumerate(train_dataloader):
            b_input_ids = batch[0].to(device)
            b_masks = batch[1].to(device)
            b_labels = batch[0].to(device)
            prompt = batch[3][0]
            original_label = batch[2][0]

            model.zero_grad()

            outputs = model(b_input_ids, labels=b_labels, attention_mask=b_masks, token_type_ids=None)
            loss = outputs[0]
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()
            with torch.no_grad():
                model.cpu()
                pred_label = classify_response_no_load(prompt, tokenizer, model)
                model.cuda()
                    
                all_preds.append(pred_label)
                all_original.append(original_label)

        avg_train_loss = total_train_loss / len(train_dataloader)
        train_acc = accuracy_score(all_original, all_preds)
        train_recall = recall_score(all_original, all_preds, average='macro', zero_division=np.nan)
        train_precision = precision_score(all_original, all_preds, average='macro', zero_division=np.nan)
        train_f1 = f1_score(all_original, all_preds, average='macro', zero_division=np.nan)

        print(f"Training Loss: {avg_train_loss}")
        print("Training Metrics:")
        print(f"Accuracy: {train_acc:.2f}")
        print(f"Recall: {train_recall:.2f}")
        print(f"Precision: {train_precision:.2f}")
        print(f"F1-Score: {train_f1:.2f}\n")

        total_train_loss_list.append(avg_train_loss)
        total_train_acc.append(train_acc)
        total_train_recall.append(train_recall)
        total_train_precision.append(train_precision)
        total_train_f1.append(train_f1)

        total_val_loss = 0
        all_preds = []
        all_original = []

        model.eval()

        for step, batch in enumerate(val_dataloader):            
            b_input_ids = batch[0].to(device)
            b_masks = batch[1].to(device)
            b_labels = batch[0].to(device)
            prompt = batch[3][0]
            original_label = batch[2][0]
            with torch.no_grad():
                outputs = model(b_input_ids, labels=b_labels, attention_mask=b_masks, token_type_ids=None)
                loss = outputs[0]
                total_val_loss += loss.item()
                model.cpu()
                pred_label = classify_response_no_load(prompt, tokenizer, model)
                model.cuda()
                
                all_preds.append(pred_label)
                all_original.append(original_label)

        avg_val_loss = total_val_loss / len(val_dataloader)
        val_acc = accuracy_score(all_original, all_preds)
        val_recall = recall_score(all_original, all_preds, average='macro', zero_division=np.nan)
        val_precision = precision_score(all_original, all_preds, average='macro', zero_division=np.nan)
        val_f1 = f1_score(all_original, all_preds, average='macro', zero_division=np.nan)

        print(f"Validation Loss: {avg_val_loss}")
        print("Validation Metrics:")
        print(f"Accuracy: {val_acc:.2f}")
        print(f"Recall: {val_recall:.2f}")
        print(f"Precision: {val_precision:.2f}")
        print(f"F1-Score: {val_f1:.2f}\n")

        total_val_loss_list.append(avg_val_loss)
        total_val_acc.append(val_acc)
        total_val_recall.append(val_recall)
        total_val_precision.append(val_precision)
        total_val_f1.append(val_f1)
        model.cpu()

    pwd = os.path.dirname(__file__)
    models_path = os.path.join(pwd, './saved_models/example_classification_training_weights')

    if not os.path.exists(models_path):
        os.makedirs(models_path)

    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(models_path)
    tokenizer.save_pretrained(models_path)

    fig, ax = plt.subplots(1, 1)
    ax_train, = ax.plot(total_train_loss_list)
    ax_val, = ax.plot(total_val_loss_list)
    ax.set_ylabel("Loss")
    ax.set_xlabel("Epochs")
    ax.legend([ax_train, ax_val], ["Training", "Validation"])
    ax.set_title("Training Loss versus Validation Loss")
    fig.savefig(os.path.join(pwd, "./figures/loss.png"))
    plt.close(fig)

    fig, ax = plt.subplots(1, 1)
    ax_train, = ax.plot(total_train_acc)
    ax_val, = ax.plot(total_val_acc)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Epochs")
    ax.legend([ax_train, ax_val], ["Training", "Validation"])
    ax.set_title("Training Accuracy versus Validation Accuracy")
    fig.savefig(os.path.join(pwd, "./figures/acc.png"))
    plt.close(fig)

    fig, ax = plt.subplots(1, 1)
    ax_prec, = ax.plot(total_train_precision)
    ax_rec, = ax.plot(total_train_recall)
    ax.set_ylabel("Precision/Recall")
    ax.set_xlabel("Epochs")
    ax.legend([ax_prec, ax_rec], ["Precision", "Recall"])
    ax.set_title("Training Precision versus Recall")
    fig.savefig(os.path.join(pwd, "./figures/train_prec_rec.png"))
    plt.close(fig)

    fig, ax = plt.subplots(1, 1)
    ax_prec, = ax.plot(total_val_precision)
    ax_rec, = ax.plot(total_val_recall)
    ax.set_ylabel("Precision/Recall")
    ax.set_xlabel("Epochs")
    ax.legend([ax_prec, ax_rec], ["Precision", "Recall"])
    ax.set_title("Validation Precision versus Recall")
    fig.savefig(os.path.join(pwd, "./figures/val_prec_rec.png"))
    plt.close(fig)

    fig, ax = plt.subplots(1, 1)
    ax_train, = ax.plot(total_train_f1)
    ax_val, = ax.plot(total_val_f1)
    ax.set_ylabel("F1-Score")
    ax.set_xlabel("Epochs")
    ax.legend([ax_train, ax_val], ["Training", "Validation"])
    ax.set_title("Training F1-Score versus Validation F1-Score")
    fig.savefig(os.path.join(pwd, "./figures/f1-score.png"))
    plt.close(fig)



# run file by itself to train weights
if __name__ == "__main__":
    while True:
        try:
            response = int(input("1 to train model for generation, 0 to train model for classification: "))
            if response == 0 or response == 1:
                break
            raise ValueError
        except ValueError:
            print("Please input a valid option")
    
    if response == 1:
        train_model()
        print(generate_response("Question: Where is the key? Answer: "))
    else:
        train_model_classification()
        print("Question: Where is the key?")
        print(classify_response("Where is the key?"))
        print("Question: Where am I?")
        print(classify_response("Where am I?"))
        print("Question: Who are you?")
        print(classify_response("Who are you?"))
        print("Question: How is the weather?")
        print(classify_response("How is the weather?"))
