from gensim.models import keyedvectors
import pandas as pd 
import time

address = "C:\Users\t-yual\Desktop\dataset\wiki.en.vec"
def Loading_to_memory(file_address):
    
    print("Loading begins...")
    start_time = time.time()
    en_model = keyedvectors._load_word2vec_format(file_address)
    end_time = time.time()
    print("loading successful !")
    print("{} seconds have elapsed".format(start_time-end_time))
    return en_model

en_model = Loading_to_memory(address)

