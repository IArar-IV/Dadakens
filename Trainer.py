

import torch
import torch.utils.data as data
import torch.nn as nn
import model as m
import Dataset as TheOnlyAndRealDataAdrisSetOfTweetsWeAreGoingToWorkWithCarlosCanIGetTheNPass
import matplotlib.pyplot as plt
import pickle

dataset = TheOnlyAndRealDataAdrisSetOfTweetsWeAreGoingToWorkWithCarlosCanIGetTheNPass.HSDataset('trainTweets4.csv')

proletarians = 6

batch_size = 1

train_loader = data.dataloader.DataLoader(dataset, num_workers = proletarians)

datatest = TheOnlyAndRealDataAdrisSetOfTweetsWeAreGoingToWorkWithCarlosCanIGetTheNPass.HSDataset('testTweets4.csv', offset=(1 + len(list(dataset.vocabulary_idx))))

test_loader = data.dataloader.DataLoader(datatest, num_workers = proletarians)



def vocabulary(d1, d2):
    set_ = set(list(d1.vocabulary_idx) + list(d2.vocabulary_idx))
    return set_

voc_len = len(vocabulary(dataset, datatest))

print('total voc: ', voc_len)

network = m.IARar_IV(voc_len, 50, dataset.getCtxSize())

loss_f = nn.BCELoss()

LR = 0.00003

optimizer = torch.optim.RMSprop(network.parameters(), lr = LR)

schedude = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 1700, gamma = 0.9)

pickle.dump(dataset.getDict(), open('./transTrain.p', 'wb'))
pickle.dump(datatest.getDict(), open('./transTest.p', 'wb'))

def goSAF(network, epoches):
    
    loss_values = []
    loss_plot = []
    
    for epoch in range(epoches):
        counter = 0
        
        for x, y in train_loader:
            
            #x, y = dataset[i]
            
            prediction = network(x)
            loss = loss_f(prediction.view(-1), y.view(-1))
            loss_values.append(loss.item())
            loss.backward()

            
            if counter % 70 == 0:
                optimizer.step()
                network.zero_grad()
                torch.save(network.state_dict, './statedictnet.p')
                
                mitj = sum(loss_values)/len(loss_values)
                loss_plot.append(mitj)
                loss_values = []
                plt.plot(loss_plot)
                plt.savefig("./tmpPlot.png")
                plt.clf()
            counter += 1
            schedude.step()
        
        for x, y in test_loader:
            
            counter += 1
            #x, y = dataset[i]
            prediction = network(x)
            loss = loss_f(prediction, y)
            loss_values.append(loss.item())
            if counter % 50 == 0:
                mitj = sum(loss_values)/len(loss_values)
                loss_plot.append(mitj)
                loss_values = []
                plt.plot(loss_plot)
                plt.savefig("./tmpPlotTest.png")
                plt.clf()
                
            counter += 1

if __name__ == '__main__':
    
    goSAF(network, 100)
            
            
            
            
