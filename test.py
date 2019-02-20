"""import pyodbc as db
import numpy as np 
from store_and_restore import Change_format
connection = db.connect("Driver={SQL Server Native Client 11.0};"
			            "Server=(localdb)\mssqllocaldb;"
			            "Database=TESTDB;"
		                "Trusted_Connection=yes;")

cursor = connection.cursor()



from data_helpers import clean_str
import pandas as pd 

df = pd.read_csv(r"C:\Users\t-yual\Desktop\dataset\training.csv", encoding="latin-1")
content = df.iloc[0, 5]
string = clean_str(content)
string_list = string.split(" ")
sentence = []
Non_token = np.zeros((1,300))
print("The length of the list is {}".format(len(string_list)))


for x in string_list:
    print("inserting {}".format(x))
    cursor.execute("select vector from EnWordVec where word = ?", x)
    row = cursor.fetchone()
    if row ==None:
        print("Can't find {} in the database !".format(x))
        sentence = sentence.append(Non_token)
    sentence.append(Change_format(row))

sentence = np.array(sentence)
print("The shape of the list is {}".format(np.shape(sentence)))

"""
from sql_import import Loading_to_memory
from data_helpers import clean_str
import numpy as np
import pandas as pd 
import time 
import os
address = "C:\Users\t-yual\Desktop\dataset"
en_model = Loading_to_memory(os.path.join(address, "wiki.en.vec"))
df = pd.read_csv(os.path.join(address, "training.csv"), encoding="latin-1")



def enumerate_append(remaining_list):

    try:
        for i, word in enumerate(remaining_list):
            



test_string = clean_str(df.iloc[0, 5])
string_list = test_string.split(" ")
sentence = []
start_time = time.time()
for i, word  in enumerate(string_list):
    try:
        sentence.append(en_model[x])
    except Exception:
        sentence.append(np.zeros((1,300),type=float))




################################################################


a = [1,2,0,3,4]
try:
    for i, item in enumerate(a):
        print(2/item)
except Exception:
    print("{} by zero".format(i))
    iterate_append(a[i:])

def iterate_append(test_list):

    try:
        for i, item in enumerate(a):
            print(2/item)
    except Exception:
        print("{} by zero".format(i))
        iterate_append(a[i:])

