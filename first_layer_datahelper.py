import pandas as pd 
import numpy as np 
from data_helpers import enumerate_append, debug_enumerate_append
from gensim.models import KeyedVectors
import os
import time
import sys 


def process_csv(filename, max_len):

    df = pd.read_csv(filename, encoding="latin-1")
    words = []
    scores = []
    input_x = []
    input_y = []
    print("Loading data...")
    for i in range(df.shape[0]):
        words.append(df.iloc[i, 1])
        scores.append(df.iloc[i, 2])
    print("Loading Successfully!")
    
    num_list = int((len(words)-1)/max_len) + 1
    remaining_words_num = num_list * max_len - len(words)
    remaining_words = []
    remaining_scores = []
    for i in range(remaining_words_num):
        remaining_words.append(words[i])
        remaining_scores.append(scores[i])
    new_words = words + remaining_words
    new_scores = scores + remaining_scores
    
    for i in range(num_list):
        start_index = i * max_len
        end_index = (i+1) * max_len
        input_x.append(new_words[start_index:end_index])
        input_y.append(new_scores[start_index:end_index])
    
    return [input_x, input_y]











def training_data_pay(filename_csv, filename_vec, max_len, debug=False):

    input_x, input_y = process_csv(filename_csv, max_len)
    if not debug:
        en_model = load_word2vec(filename_vec)
        input_x_vector = text_to_vector(input_x, en_model)
    else:
        input_x_vector = debug_text_to_vector(input_x)
    dev_sample_index = -1 * int(0.1 * float(len(input_x_vector)))
    train_x, dev_x = input_x_vector[:dev_sample_index], input_x_vector[dev_sample_index:]
    train_y, dev_y = input_y[:dev_sample_index], input_y[dev_sample_index:]
    return [train_x, train_y, dev_x, dev_y]















