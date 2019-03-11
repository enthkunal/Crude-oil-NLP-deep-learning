import os, json
import numpy as np
from operator import itemgetter
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from tqdm import tqdm
import keras
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

## Importing news data from csv file to system
json_path = "/Users/kunal/Documents/RIC/crude oil price /NewsData/"
sub_directories = {"Jul", "Aug", "Sep", "Oct", "Nov", "Dec"}
Data = []

##Loading articles from JSON files into python environment
def loadJsonData(sub_directories, json_path):
    for subdir in sub_directories:
        targetDir = json_path + subdir
        json_files = [pos_json for pos_json in os.listdir(targetDir) if pos_json.endswith(".json")]
        for File in json_files:
            jsonData = json.loads(open(targetDir + "/" + File, encoding='utf-8').read())
            Data.append(jsonData)
        sortedData = sorted(Data, key=itemgetter('published'))
    return sortedData;

Data = loadJsonData(sub_directories, json_path)
cnt = 0
newsDateData = list()
for obj in Data:
    newsDateData.append([obj['uuid'], obj['published'].split('T')[0], obj['title'], obj['text'], obj['thread']['site'],
                         obj['thread']['site_type'],obj['thread']['domain_rank']])
newsDateData

# To pandas DataFrame
newsDateData = pd.DataFrame(newsDateData, columns=['uuid', 'published', 'title', 'text','site','site_type','domain_rank'])


# change string to Datetime
newsDateData.published = pd.to_datetime(newsDateData.published)
newsDateData.to_csv("/Users/kunal/Documents/RIC/crude oil price /NewsData/news_csv_data.csv",
                    encoding='utf-8')
# Loading historical crude oil price
CrudeData = pd.read_csv("/Users/kunal/Documents/RIC/crude oil price /NewsData/Brent Oil Futures Historical Data.csv")
CrudeData.Date = pd.to_datetime(CrudeData.Date)


# merge articles with crude oil price data
mergedData = pd.merge(newsDateData, CrudeData,
                      how='inner',
                      right_on="Date", left_on="published", sort=True)


# sort data by article publication date
finalData = mergedData.sort_values('published')
finalData.to_csv("/Users/kunal/Documents/RIC/crude oil price /NewsData/finaldata.csv", encoding='utf-8',
                 escapechar='/')

#Exporting data to with delimated format
finalData.to_csv("/Users/kunal/Documents/RIC/crude oil price /NewsData/newdata.csv", encoding='utf-8',
                 escapechar=',')


#  preprocessing the dataset(we dont need uuid, percentage change etc...)

finalData.drop(['uuid', 'Change %','Vol.'],axis=1)









