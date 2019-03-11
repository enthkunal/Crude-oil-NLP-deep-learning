

import pandas as pd
import numpy as np
import keras

from keras import backend as K
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense
from keras.models import Sequential

# in this cell we will just define a few helper functions

def build_embedding_matrix(emb_path, emb_type, tokenizer, vocab_size):
    """This function will read the word embeddings from the "emb_path" path and will build
    embedding matrix. To speed experimenting, it will save only necessary word embeddings
    in a separate file, so that there is no need to read the large embedding files every
    time.Args:emb_path: path to the embedding file.
    emb_type: type of embeddings (could be word2vec, fasttext, glove)
        tokenizer: Keras tokenizer with data, that we want to process later (for which we will build)
                   embedding matrix.
        vocab_size: size of the vocabulary, would usually be = len(tokenizer.word_index) + 1
    Returns:
        file name of the file with the generated embedding matrix.
    """
    if emb_type == 'glove':
        emb_index = {}
        with open(emb_path, 'r') as emb_file:
            for line in emb_file:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                emb_index[word] = coefs
        
        emb_dim = emb_index['man'].shape[0]
        out_name = 'glove-{}.npy'.format(emb_dim)
    elif emb_type == 'fasttext':
        emb_index = gensim.models.KeyedVectors.load_word2vec_format(emb_path)
        
        emb_dim = emb_index['man'].shape[0]
        out_name = 'fasttext-{}.npy'.format(emb_dim)        
    elif emb_type == 'word2vec':
        emb_index = gensim.models.KeyedVectors.load_word2vec_format(emb_path, binary=True)
        emb_dim = emb_index['man'].shape[0]
        out_name = 'word2vec-{}.npy'.format(emb_dim)
    else:
        raise ValueError('emb_type = {} is not supported'.format(emb_type))
    
    # words not found in embedding index will be all zeros.
    emb_matrix = np.zeros((vocab_size, emb_dim))
    for word, word_idx in tokenizer.word_index.items():
        if word in emb_index:
            emb_matrix[word_idx] = emb_index[word]
    
    # saving embedding matrix to file
    with open(out_name, 'wb') as fout:
        np.save(fout, emb_matrix)
    
    return out_name

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 

# reading the data from file
filename = '/Users/kunal/Documents/RIC/crude oil price /NewsData/Up.csv'
data = pd.read_csv(filename, encoding='ISO-8859-1', usecols=['text', 'Price'])
# shuffling the data
data = data.sample(frac=1).reset_index(drop=True)
print("Data shape: {}".format(data.shape))


# In[4]:


data= data.astype({"text":str})


# In[5]:


# preparing data, i.e. transforming from strings to list of tokens
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['text'])
sequences = tokenizer.texts_to_sequences(data['text'])
# adding padding to maximal length, i.e. filling the end of the 
# sequences with zeros, so that they are all the same length and 
# can be used in one batch together
padded_sequences = pad_sequences(sequences)
# word_index is the mapping from words to integer indexes that Tokenizer built
# adding +1 for out-of-vacabulary words
vocab_size = len(tokenizer.word_index) + 1

import gensim

all_embeddings = [
    ('/Users/kunal/Documents/RIC/crude oil price /NewsData/Embeddings/glove.6B/glove.6B.300d.txt', 'glove'), 
    ('/Users/kunal/Documents/RIC/crude oil price /NewsData/Embeddings/GoogleNews-vectors-negative300.bin.gz', 'word2vec'),
    ('/Users/kunal/Documents/RIC/crude oil price /NewsData/Embeddings/wiki-news-300d-1M.vec', 'fasttext'),
]
for emb_path, emb_type in all_embeddings:
    build_embedding_matrix(emb_path, emb_type, tokenizer, vocab_size)


def build_conv_network(emb_matrix_path, input_length):
    """This function builds convolutional network to solve the prediction problem.
    Args:
        emb_matrix_path: path to the pretrained embedding matrix 
                         (generated with `build_embedding_matrix` function)
        input_length: length of input sequences (should be = padded_sequences.shape[1])
    Returns:
        compiled Keras model
    """
    # reading the embedding matrix
    with open(emb_matrix_path, 'rb') as fin:
        emb_matrix = np.load(fin)
    
    # building the model
    model = Sequential()
    model.add(Embedding(emb_matrix.shape[0], emb_matrix.shape[1], weights=[emb_matrix], 
                        trainable=False, input_length=input_length))
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(MaxPooling1D(5))
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(MaxPooling1D(20))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(lr=0.001), metrics=[rmse])
    print(model.summary())
    return Model
# Buiding model based on Fastext embedding.
model = build_conv_network('fasttext-300.npy', padded_sequences.shape[1])
validation_split = 0.2  # taking 1/5 of the data for the validation
epochs = 50
batch_size = 100
history = model.fit(padded_sequences, data['Price'], validation_split=validation_split,  
          epochs=epochs, batch_size=batch_size)
# Buiding model based on Word2vec embedding.
model = build_conv_network('word2vec-300.npy', padded_sequences.shape[1])
validation_split = 0.2   # taking 1/5 of the data for the validation
epochs = 50
batch_size = 128
history = model.fit(padded_sequences, data['Price'], validation_split=validation_split,  
          epochs=epochs, batch_size=batch_size)

# Buiding model based on Glove embedding.
model = build_conv_network('glove-300.npy', padded_sequences.shape[1])
validation_split = 0.2   # taking 1/5 of the data for the validation
epochs = 50
batch_size = 128
history = model.fit(padded_sequences, data['Price'], validation_split=validation_split,  
          epochs=epochs, batch_size=batch_size)

# Model Loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Glove Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#model Performance
plt.plot(history.history['rmse'])
plt.plot(history.history['val_rmse'])
plt.title('Glove Model performance')
plt.ylabel('rmse')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

pd.pandas.DataFrame(history1.history).to_csv("/Users/kunal/Desktop/Word Embedding/cnn/history.csv")
