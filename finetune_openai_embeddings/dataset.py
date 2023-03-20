import torch
from torch.utils.data import Dataset, DataLoader

class RetrievalDataset(Dataset):
    def __init__(self, embeddings, query_pairs):
        self.length = len(query_pairs)
        self.query_pairs = query_pairs
        self.embeddings = embeddings
        
    def __getitem__(self, index):
        query_pair = self.query_pairs[index]
        query, doc = query_pair['query'], query_pair['doc']
        query_emb = self.embeddings[query]
        doc_emb = self.embeddings[doc]
        label = torch.tensor(query_pair['label'])
        return query_emb, doc_emb, label
    
    def __len__(self):
        return self.length
    