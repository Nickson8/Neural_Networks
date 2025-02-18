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


#Standardization of X
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)

X_standardized = (X - mean) / std


model = LSTM(32, 1)

model.fit(X_standardized, Y)

model.train(lr=0.01, interations=60)

#model.predict(Xzao=X[:300], Yzao=Y[:300])