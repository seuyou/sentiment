import numpy as np
import re
import pandas as pd
import time
from gensim.models import KeyedVectors
import sys

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", "\'s", string)
    string = re.sub(r"\'ve", "\'ve", string)
    string = re.sub(r"n\'t", "n\'t", string)
    string = re.sub(r"\'re", "\'re", string)
    string = re.sub(r"\'d", "\'d", string)
    string = re.sub(r"\'ll", "\'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"won\'t", "will not", string)
    string = re.sub(r"\'s", " is", string)
    string = re.sub(r"\'re", " are", string)
    string = re.sub(r"\'m", " am", string)
    string = re.sub(r"n\'t", " not", string)
    string = re.sub(r"\'ll", " will",string)
    string = re.sub(r"[\"\']", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    
    return string.strip().lower()



def extract_words(string):

    string = re.sub(r"[#][0-9]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()



def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r", encoding='utf-8').readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r", encoding='utf-8').readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


def batch_iter(data_x, data_y, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    Before processing by this function, you data_x should be of the same length for each sentence
    """
    
    data_x = np.array(data_x)
    data_y = np.array(data_y)
    data_size = len(data_x)
    num_batches_per_epoch = int((len(data_x)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data_x, shuffled_data_y = data_x[shuffle_indices], data_y[shuffle_indices]
        else:
            shuffled_data_x, shuffled_data_y = data_x, data_y
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield [shuffled_data_x[start_index:end_index], shuffled_data_y[start_index:end_index]]



def enumerate_append(remaining_list, en_model):
    # Change the text words into vectors that from pre-trained vectors
    sentence_vector = []
    for word in remaining_list:
        try: 
            word_vector = np.array(en_model[word])
            sentence_vector.append(word_vector)
        except Exception:
            sentence_vector.append(np.zeros((300, )))
    return sentence_vector
          


def debug_enumerate_append(remaining_list):
    # Change the text words into vectors for debug purpose
    sentence_vector = []
    for word in remaining_list:
        sentence_vector.append(np.zeros(300, ))
    
    return sentence_vector



def format_to_same_length(training_data, maxlen):
    '''
        read the sentence_polarity training data from csv file
        and change the sentences of different lengths into those of the same length
    '''
    print("reading training data from:{}, waiting...".format(training_data))
    df = pd.read_csv(training_data, encoding="latin-1")
    sentences = []
    polarities = []
    for i in range(df.shape[0]):
        sentences.append(df.iloc[i, 1].split(" "))
        polarities.append(df.iloc[i, 2])
    print("Reading successfully")   
    print("The length of sentences before padding is:{}".format(len(sentences)))
    for i, sentence in enumerate(sentences):
        if len(sentence) > 80:
            sentences[i] = sentence[:80]
        elif len(sentence) < 80:
            sentence = sentence + ["0"] * (80 - len(sentence))
            sentences[i] = sentence
    return [sentences, polarities]
    

def training_data_pay(input_x, y, word2vec):
   
    input_x_vector = text_to_vector(input_x, word2vec)
    input_y_vector = []
    for i in range(len(y)):
        if y[i] == 0:
            input_y_vector.append([1, 0])
        else:
            input_y_vector.append([0, 1])
  
    return [input_x_vector, input_y_vector]
    

def text_to_vector(text, filename):
    "Before processing by this function, the sentences the text contains should have the same length"
    "Change the data from text words to pretrained word vectors"
    try:
        print("Loading word2vec from:{}".format(filename))
        en_model = KeyedVectors.load_word2vec_format(filename)
        print("Loading successfully!")
    except Exception:
        print("Loading failed!")
        sys.exit()
    vector = []
    for batch in text:
        vector.append(enumerate_append(batch, en_model))
    return vector


    
def debug_text_to_vector(text):

    vector = []
    for batch in text:
        vector.append(debug_enumerate_append(batch))

    return vector



    



