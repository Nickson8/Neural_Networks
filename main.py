from lstm import LSTM
from data import plot_eeg, load_arff
import pandas as pd
import numpy as np

# plot_eeg()

df = load_arff('filtered_output.arff')

X = df.iloc[:, :-1].to_numpy()
y = df.iloc[:, -1].to_numpy(dtype=int)

#One-Hot Enconding y
Y = np.eye(2)[y]

model = LSTM(32, 1)

model.fit(X, Y)