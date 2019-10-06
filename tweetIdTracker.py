import twitter
import csv
import pandas as pd
import numpy as np

def getTweetsById(consumerKey, consumerSecret, accesToken, accesTokenSecret, finalCSV, readCSV, badWordsIds, goodWordsIds):
    api = twitter.Api(consumer_key= consumerKey,
                  consumer_secret=consumerSecret,
                  access_token_key=accesToken,
                  access_token_secret=accesTokenSecret)
    
    csvWriter = csv.writer(open(finalCSV, 'a+'))
    csvWriter.writerow(["Groundtruth", "id"])
    
    dataset = np.array(pd.read_csv(goodWordsIds))
    datasetBad = np.array(pd.read_csv(badWordsIds))
    
    correctOnes = 0
    correctGoodOnes = 0
    itera = 0
    connections = 0
    for i in dataset:
        if(itera%15):# 15 are the maximum allowed connections for twitter
            connections += 1
            del api
            api = twitter.Api(consumer_key=consumerKey,
                          consumer_secret=consumerSecret,
                          access_token_key=accesToken,
                          access_token_secret=accesTokenSecret)
        try:
            itera+=1
            a = api.GetStatus(str(i[0])).text
            csvWriter.writerow([False, a])
            correctGoodOnes += 1
        except:
            csvWriter.writerow([None, "NotFound/AccountSuspended"])
            continue
    itera = 0
    for i in datasetBad:
        if(itera%15):
            connections += 1
            del api
            api = twitter.Api(consumer_key=consumerKey,
                          consumer_secret=consumerSecret,
                          access_token_key=accesToken,
                          access_token_secret=accesTokenSecret)
        try:
            itera+=1
            a = api.GetStatus(str(i[0])).text
            csvWriter.writerow([True, api.GetStatus(str(i[0])).text])
            correctOnes += 1
        except:
            csvWriter.writerow([None, "NotFound/AccountSuspended"])
            continue
        
