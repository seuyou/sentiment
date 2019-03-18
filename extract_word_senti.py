from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk import sent_tokenize, word_tokenize, pos_tag
import pandas as pd 
import os


csv_address = r"C:\Users\liuyuan\Desktop\sentiment\dataset\training_data\training_data.csv"
words_scores_csv =  r"C:\Users\liuyuan\Desktop\sentiment\dataset\training_data\words_scores.csv"

def penn_to_wn(tag):
    """
    Convert between the PennTreebank tags to simple Wordnet tags
    """
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    return None

def mapping_senti_words(csv_address, words_scores_csv):
    df = pd.read_csv(csv_address, encoding="latin-1")
    sentences = []
    words_scores = []
    for i in range(df.shape[0]):
        sentences.append(df.iloc[i, 1])


    for sentence in sentences:
        tagged_sentence = pos_tag(word_tokenize(sentence))
        for word, tag in tagged_sentence:
            wn_tag = penn_to_wn(tag)
            if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
                continue
            synsets = wn.synsets(word, pos=wn_tag)
            if len(synsets) == 0:
                continue 
                    
            synset = synsets[0]
            swn_synset = swn.senti_synset(synset.name())
            sentiment = swn_synset.pos_score() - swn_synset.neg_score()
            words_scores.append([word, sentiment])

    columns = ["word", "score"]
    to_write = pd.DataFrame(data=words_scores, columns=columns)

    to_write.to_csv(words_scores_csv)