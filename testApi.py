# from raw_cnn.text_cnn import TextCNN
# import tensorflow as tf 
# import numpy as np 
# input_x = np.ones((1, 80, 300))
# input_y = np.ones((1,2))
# cnn = TextCNN(80, 2, 300, [2, 3, 4], 3)
# init = tf.global_variables_initializer()
# sess = tf.Session()
# sess.run(init)
# sess.run(cnn.accuracy, feed_dict={cnn.input_x:input_x, cnn.input_y:input_y})
# from train import preprocess
# x_train, y_train, x_dev, y_dev = preprocess()
# print("Successful")
# import matplotlib.pyplot as plt 
# import pandas as pd 
# import seaborn as sns 
# from data_helpers import clean_str
# training_data_file = r"C:\Users\liuyuan\Desktop\sentiment\dataset\training_data\training_data.csv"
# df = pd.read_csv(training_data_file, encoding="latin-1")
# length = [len(clean_str(df.iloc[i, 1]).split(" ")) for i in range(df.shape[0])]
# sns.set(color_codes=True)
# sns.distplot(length)
# max_len = max(length)
# min_len = min(length)
# print(min_len)
# print("The averge length of all the sentences is {}".format(sum(length)/len(length)))
# plt.show()
import pandas as pd
from data_helpers import clean_str
import os
neg_add = r"C:\Users\liuyuan\Desktop\sentiment\cnn-text-classification-tf\data\rt-polaritydata\rt-polarity.neg"
pos_add = r"C:\Users\liuyuan\Desktop\sentiment\cnn-text-classification-tf\data\rt-polaritydata\rt-polarity.pos"
csv_add = r"C:\Users\liuyuan\Desktop\sentiment\dataset\training_data\pos_neg.csv"
f_neg = list(open(neg_add, "r", encoding="utf-8").readlines())
f_pos = list(open(pos_add, "r", encoding="utf-8").readlines())
data = []

for line in f_neg:
    data.append([clean_str(line), 0])
for line in f_pos:
    data.append([clean_str(line), 1])
columns = ["sentence", "polarity"]
df = pd.DataFrame(data=data, columns=columns)
df.to_csv(csv_add)







    




