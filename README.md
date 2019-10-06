####README####

IArar-IV Net. LSTM Classifier for hate speech detection.

Implements an algorithm of hate-speech dtection that, unlike other common alternatives, learns from the full context of a message.

Docs:

 0 - architecture.odp: Documentates the network architecture.

Modules:
 1 - CleanTweet.py: Given a Tweet as entry; cleans and translate the Tweet.
 2 - model.py: Implements the network architecture in pytorch.
 3 - Dataset.py: Implements a 'Dataset' object for parsing and using the csv datasets.
 4 - Trainer.py: Given the datasets and the model trains a network so it can classify between hate/non hate tweet speeches.

Data:

We've obtained some ammount of data. We cleaned all the Tweets and setted their labels attending to different parameters specified in every dataset downloaded. Here are listed both dataset training and testing obtained from many other smalls datasets.

 5 - trainTweets4.csv: > 40000 positive and negative labeled tweets.
 6 - testTweets4.csv: > 2000 positive and negative labeled tweets.

Savestates:
 
A few savestates from the training script.

 7 - statedictnet.p: Pickled state dictionary of the network for k-iteration.
 (https://drive.google.com/file/d/18kDhUishr-1gywMaMD1qtcZbSB4iD9bh/view?usp=sharing)
 8 - transTrain.p: Pickled dictionary with the encoding of the training set vocabulary.
 9 - transTest.p: Pickled dictionary with the encoding of the test set vocabulary.
 10 - tmpPlot.png: Plotted training of the first k-iterations. We didn't have enought time for iterate over further epoches.

Extra:
 
 11 - TMP.py: Module used for cleaning a negative-labeled Tweets dataset.
 12 - getCompleteCsv.py: Module used for merging a few datasets we cleaned.
 13 - tweetIdTRacker.py: We tried scrapping Twitter from an IDs dataset but they don't allow you to get more than a limited number of requests so it was not useful at all because of the time spending.


