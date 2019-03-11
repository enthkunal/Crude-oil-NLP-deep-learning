

import pandas as pd
import numpy as np
### Importing aggregated data

df = pd.read_csv('/Users/kunal/Documents/RIC/crude oil price /NewsData/Classification/newdata.csv',encoding='ISO-8859-1')

#### Basic feature extraction from text data
#Dropping unnessary columns


list(df)


df.drop(['Unnamed: 0','uuid','Date','title','Vol.'],axis=1, inplace=True)

# Now the unwanted columns are gone, so it's time to check what are basic features in the text column
#counting number of words

df['number_of_words'] = df.text.apply(lambda x: len(x.split()))
#number of charecters
df['char_count'] = df['text'].str.len()


#It seems that there are lot of white spaces included in the text column

def avg_word(sentence):
    words = sentence.split()
    while len(words) != 0:
        return (sum(len(word) for word in words)/len(words))
df['avg_word'] = df['text'].apply(lambda x: avg_word(x))


# Counting number of stopwords, with the help of NLTK natural language toolkit
from nltk.corpus import stopwords
stop = stopwords.words('english')

df['stopwords'] = df['text'].apply(lambda x: len([x for x in x.split() if x in stop]))

#counting number of upper case charecters:
df['upper'] = df['text'].apply(lambda x: len([x for x in x.split() if x.isupper()]))



#now the raw text in the Text column needs to be processed, we will be using NLTK toolkit to process.
#### Data Preprocessing with NLTK  natural language toolkit
#Transforing the text column into lower case letters

df['text'] = df['text'].apply(lambda x: " ".join(x.lower() for x in x.split()))

# removing numbers from text column
from string import digits
df['text'] = df['text'].str.replace('\d+', '')

#Removing the punctuations in text column

df['text'] = df['text'].str.replace('[^\w\s]','')

#removing stopwords from text column

from nltk.corpus import stopwords

stop = stopwords.words('english')

df['text'] = df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

# checking for more frequent words in text column

freq = pd.Series(' '.join(df['text']).split()).value_counts()[:10]

#checking for rare words in text column


rare_freq = pd.Series(' '.join(df['text']).split()).value_counts()[-10:]

rare_freq


#since these words make no sense

rare_freq = list(rare_freq.index)
df['text'] = df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in rare_freq))

# Now there may be some comments in the dataset which needs spelling correction, correcting spelling with textblob 

# conda install -c conda-forge textblob, needs to be run on terminal


from textblob import TextBlob

df['text'][:5].apply(lambda x: str(TextBlob(x).correct()))

df.text

#Tokenization, since we already used textblob no need for tokenization 

# Stemming removing suffixes like 'ing','ly','ies','s' etc , with NLTK

from nltk.stem import PorterStemmer
st = PorterStemmer()
df['text'][:5].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))

# Result shows that stemming does not consider the meaning of the word and it just cut down the suffix part, better option is 
#to use lemanization over stemmming. Since stemming converts the word to its root word.

import nltk
nltk.download('wordnet')
from textblob import Word

df['text'] = df['text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

stop_words = set(stopwords.words('english'))
stop_words.update(['.', ',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}', '\\', '/',
                   '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])

df['text'] = df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop_words))

# Removing all foreign charecters which are not removed by UTF-8 encoding

df['text'] = df['text'].apply(''.join).str.replace('[^A-Za-z\s]+', '')

df.to_csv("/Users/kunal/Documents/RIC/crude oil price /NewsData/Up.csv")



