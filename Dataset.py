import torch.utils.data as Data
import pandas as pd
import numpy as np
import torch

class HSDataset(Data.Dataset):
    
    def __init__(self, csvFilename, context_size = 0, offset = 0):
        
        
        self.off = offset
        #Array tipus [Tweet, GT]
        
        self.array = np.array(pd.read_csv(csvFilename, encoding = 'latin-1'))
        
        self._list = []
        
        for sample in self.array:
            
            self._list.append([sample[0], sample[1]])
            
        self.list = np.array(self._list)
        llista_paraules = []
        for sample in self.list:
            for word in sample[0].split():               
                if not word in llista_paraules:
                    llista_paraules.append(word)
                    
        self.vocabulary_idx = {i: j+1+self.off for j, i in enumerate(llista_paraules)}
        
        self.context_size = self.getCtxSize()
        
        print('Vocabulary: ', len(llista_paraules))
        print('Samples: ', len(self.list))
        
    def getDict(self):
        return self.vocabulary_idx
    
    def getCtxSize(self):
        return max([len(x[0].split()) for x in self.list])
    
    def __len__(self):
        
        return len(self._list)
    
    def toTensor(self, boolean):
        
        if not boolean in ['False', False]:
            
            return torch.tensor([0., 1.], dtype = torch.float32)

        else:
            
            return torch.tensor([1., 0.], dtype = torch.float32)
    def make_context_vector(self, context, word_to_ix):
        
        idxs = [word_to_ix[w] for w in context]
        return torch.tensor(idxs, dtype=torch.long)
    
    def __getitem__(self, index):
        
        
            text, boolean = self.list[index]
            
            Y = self.toTensor(boolean)
            
            x = [0 for x in range(self.context_size)]
            listed = text.split()
            for k, word in enumerate(listed):
                x[k+len(x)-len(listed)] = self.vocabulary_idx[word] #So it is a seq [-1, -1 ...., words]
                
            X = torch.tensor(x, dtype=torch.long)
            
            return X, Y
                
                
            
    
            
            