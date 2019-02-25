import numpy as np
import re
import pandas as pd

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
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"can\'t", "can not", string)
    string = re.sub(r"won\'t", "will not", string)
    string = re.sub(r"shouldn\'t", "should not",string)
    string = re.sub(r"mustn\'t", "must not", string)
    string = re.sub(r"isn\'t", "is not", string)
    string = re.sub(r"aren\'t", "are not", string)
    string = re.sub(r"wasn\'t", "was not", string)
    string = re.sub(r"weren\'t", "were not", string)
    string = re.sub(r"hasn\'t", "has not", string)
    string = re.sub(r"haven\'t", "have not", string)
    string = re.sub(r"hadn\'t", "had not", string)
    string = re.sub(r"he\'s", "he is", string)
    string = re.sub(r"she\'s", "she is", string)
    string = re.sub(r"you\'re", "you are", string)
    string = re.sub(r"we\'re", "we are", string)
    string = re.sub(r"i\'m", "i am", string)
    string = re.sub(r"i\'ll", "i will", string)
    string = re.sub(r"she\'ll", "she will", string)
    string = re.sub(r"he\'ll", "he will", string)
    string = re.sub(r"you'll", "you will", string)
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


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def tokenization(string):

    string = string.strip(" ").lower()
    clean_string = clean_str(string)
    return clean_string.split(" ")


def load_data_and_labels_csv(mixed_data_file):

    "Loading the mixed polarizd data from csv file, split the data into data and labels"

    df = pd.read_csv(mixed_data_file, encoding="latin-1")
    input_x = [tokenization(df.iloc[i,5]) for i in range(df.shape[0]) ]
    input_y = []
    for i in range(df.shape[0]):
    # Change integer to list
        if df.iloc[i, 0]==0:
            input_y.append([1., 0., 0.])
        elif df.iloc[i, 0]==2:
            input_y.append([0., 1., 0.])
        else:
            input_y.append([0., 0., 1.])
    return [input_x, input_y]


def enumerate_append(remaining_list, sentence_vector, en_model):

    for i, word in enumerate(remaining_list):
        try: 
            sentence_vector.append[en_model[word]]
        except Exception:
            sentence_vector.append[np.zeros((1,300))]
            enumerate_append(remaining_list[i+1:], sentence_vector, en_model)


def change_to_vector(en_model, sentence_list, max_len):

    sentence_vector = []
    enumerate_append(sentence_list, sentence_vector, en_model)
    sentence_vector = np.array(sentence_vector)
    diff_len = max_len - np.shape(sentence_vector)[0]
    sentence_vector = np.concatenate([sentence_vector,np.zeros((diff_len, 300))], 0)
    return sentence_vector


    




    



