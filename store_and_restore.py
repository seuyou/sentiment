from gensim.models import KeyedVectors
import time
import pyodbc as db


def store_from_mem_to_sql(file_address = "wiki.en.vec", inspection = False):

    try:
        print("Loading the word vectors to main memory...")
        start_time = time.time()
        en_model = KeyedVectors.load_word2vec_format(file_address)
        end_time = time.time()

        print("Loading successfully !")
        print("{} seconds have elapsed!", end_time-start_time)

    except:
        print("Loading failed !")

    try:
        print("Begin connection to database...")
        connection = db.connect("Driver={SQL Server Native Client 11.0};"
			                    "Server=(localdb)\mssqllocaldb;"
			                    "Database=TESTDB;"
			                    "Trusted_Connection=yes;")
        print("Connecting successfully !")

    except:
        print("Connecting failed !")

    cursor = connection.cursor()
    
    print("Transfering data...")
    for word in en_model.vocab:
        
        word_vector = en_model[word]
        cast_to_string = list(map(str, word_vector))
        string = ""
        for x in cast_to_string:

            string = string+x+" "
        
        string = string.strip(" ")
        if inspection == True:
            Inspection(word)
        cursor.execute("insert into EnWordVec(word, vector) values(?, ?)", word, string)




def Inspection(word):
    
    print("Inserting {}".format(word))



def Change_format(row):

    list_row = list(row)

    return list(map(float, list_row[0].split(" ")))



        
  