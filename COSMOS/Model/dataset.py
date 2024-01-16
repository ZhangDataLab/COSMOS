# torch
from torch.utils.data import Dataset
import torch as th

# others
import pandas as pd

class WikiDataSet_simple(Dataset):
    def __init__(self, path):
        
        super(WikiDataSet_simple, self).__init__()
        self.path = path
        dataset = pd.read_pickle(self.path)
        
        self.sent_feat, self.tuple_feat = [], []
        self.labels = []
        self.source = []
        
        for data in dataset:
            self.sent_feat.append(data['sent_feat'])
            self.tuple_feat.append(data['tuple_feat'])
            self.labels.append(data['label'])
            self.source.append(data['source'])

    def __getitem__(self, i):
        return self.sent_feat[i], self.tuple_feat[i], self.labels[i], self.source[i]
    
    def __len__(self):
        return len(self.labels)
            
class WikiDataSet_bert(Dataset):
    def __init__(self, path, config):
        
        super(WikiDataSet_bert, self).__init__()
        self.path = path

        dataset = pd.read_pickle(self.path)
        
        self.texts = []
        self.labels = []
        self.source = []
        self.tokenizer = config.tokenizer
        self.max_length = config.max_length
        
        for data in dataset:
            text = data["tuple_str"] + " " + data["sent_str"]
            self.texts.append(text)
            self.labels.append(data['label'])
            self.source.append(data['source'])

    def __getitem__(self, i):
        text = self.texts[i]
        label = self.labels[i]
        source = self.source[i]
        encoding = self.tokenizer(text, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
        return {'input_ids': encoding['input_ids'].flatten(), \
            'attention_mask': encoding['attention_mask'].flatten(), \
                'label': th.tensor(label), 'source': source}
    
    def __len__(self):
        return len(self.labels)