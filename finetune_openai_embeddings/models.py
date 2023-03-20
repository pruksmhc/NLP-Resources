from torch import nn
class DocumentEmbedder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, out_dim, kernel_size=3, cnn_depth=5):
        super(DocumentEmbedder, self).__init__()       
        self.layers = nn.Sequential(
            nn.Linear(1536, 256),
            nn.LeakyReLU(),
            nn.BatchNorm1d(256),
#             nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=0.5)
            )
        
    def forward(self, user_emb):
        user_emb = self.layers(user_emb)
        return user_emb

class QueryEmbedder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, out_dim):
        super(QueryEmbedder, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(1536, 256),
            nn.LeakyReLU(),
            nn.BatchNorm1d(256),
            # nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=0.5)
            )
        
    def forward(self, item_emb):
        item_emb = self.layers(item_emb)
        return item_emb

class SimilarityModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, out_dim):
        super(SimilarityModel, self).__init__()
        self.query_emb = QueryEmbedder(embedding_dim, hidden_dim, out_dim)
        self.doc_emb = DocumentEmbedder(embedding_dim, hidden_dim, out_dim)
        self.fc1 = nn.Linear(out_dim*2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.score_layer = nn.Linear(out_dim, 1)
        self.dropout = nn.Dropout(p=0.5)
    
    def forward(self, user_emb, item_emb):
        user_emb = self.query_emb(user_emb)
        item_emb = self.doc_emb(item_emb)
        combined = item_emb * user_emb
        x = self.score_layer(combined)
        return x
