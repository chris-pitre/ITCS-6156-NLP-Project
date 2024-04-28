from torch.utils.data import Dataset
import torch

class DialogueDataset(Dataset):
    def __init__(self, data, tokenizer, gpt2_type="gpt2", max_length=768):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.attention_masks = []

        for line in data:
            encodings_dict = tokenizer(line[0], truncation=True, max_length=max_length, padding="max_length")

            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attention_masks.append(torch.tensor(encodings_dict['attention_mask']))

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_masks[idx]
    
class ClassificationDataset(Dataset):
    def __init__(self, data, tokenizer, gpt2_type="gpt2", max_length=768):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.attention_masks = []
        self.labels = []

        label_map = {
            0: "key",
            1: "door",
            2: "location",
            3: "identity",
            4: "small_talk",
            5: "goal"
        }

        for line in data:
            line_text = line[0]
            line_class = int(line[1])
            prepared_text = f'<bos>Question: {line_text}\nClass: {label_map[line_class]}<eos>'
            encodings_dict = tokenizer(prepared_text, truncation=True, max_length=max_length, padding="max_length")

            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attention_masks.append(torch.tensor(encodings_dict['attention_mask']))
            self.labels.append(label_map[line_class])

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_masks[idx], self.labels[idx]