import pandas as np
from data_helpers import load_data_and_labels_csv
import numpy as np
csv_address = r"C:\Users\t-yual\Desktop\dataset\training.csv"
input_x, input_y = load_data_and_labels_csv(csv_address)
input_x = np.array(input_x)
input_y = np.array(input_y)
print("The shape of input_x: {}".format(np.shape(input_x)))
print("The shape of input_y: {}".format(np.shape(input_y)))
