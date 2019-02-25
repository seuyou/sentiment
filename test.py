import re
import pandas as pd

def extract_words(string):

    string = re.sub(r"[#][0-9]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


address = r"C:\Users\liuyuan\Desktop\sentiment\dataset\SentiWordNet.txt"
lines = open(address, "r").readlines()
clean_lines = [lines[i] for i, x in enumerate(lines) if "#" not in x[0]]
clean_list = [string.split("\t") for string in clean_lines]
print(clean_list[108])



data = []
index = []

for i, line in enumerate(clean_list):
    for word in extract_words(line[4]).split(" "):
        try:
            if not re.search(r"[0-9]", word):
                if not re.search(r"_", word) and not re.search(r"-", word):
                    data.append([word, float(clean_list[i][2])-float(clean_list[i][3])])
                else:
                    continue
            else:
                continue
        except Exception:
            print("exception")
            print("The line is {}".format(i))
        
index = list(range(len(data)))
columns = ["word", "score"]

df = pd.DataFrame(data ,index=index, columns=columns)
filename = r"C:\Users\liuyuan\Desktop\sentiment\dataset\WordSentiScore.csv"
df.to_csv(filename, encoding="utf-8")
print("Successful")



    
     
        

            



