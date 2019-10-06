import cleanTweet as CTM
import pandas
import numpy as np
import csv
import time

writer = csv.writer(open('./datadri_v1.csv', 'w'))
data = np.array(pandas.read_csv('./AdriDataset.csv'))
lens = []
print(len(data))
for i in data:
    time.sleep(0.1)
    c = CTM.tweet_cleaner(i[0])
    print(c)
    if c:
        print(c)
        writer.writerow([c, i[1]])
        lens.append(len(c))
        if len(lens)%50 == 0:
            print(c, len(lens))
