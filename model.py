import torch
import torch.nn as nn

torch.manual_seed(456789)

class IARar_IV(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        
        self.ctx = context_size
        super(IARar_IV, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, 1024)
        self.linear1 = nn.Linear(1024 * context_size, 512)
        self.linear2 = nn.Linear(512, 128)
        self.linear3 = nn.Linear(128, 2)
        
        self.softmax = nn.Softmax(1)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p = 0.25)

    def forward(self, inputs):
        
        inputs = inputs.view(-1)
        embeds = self.embeddings(inputs)
        lstm_out, _ = self.lstm(embeds.view(len(inputs), 1, -1))
        out = self.drop(lstm_out.view(1, -1))
        out = self.relu(self.linear1(out))
        out = self.relu(self.linear2(out))
        out = self.softmax(self.linear3(out))
        
        return out
    
if __name__ == '__main__':
    
    print(IARar_IV(10, 4, 4)(torch.tensor([1, 0, 2, 3], dtype = torch.long)))