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
