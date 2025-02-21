from lstm import LSTM
from data import plot_eeg, load_arff
import pandas as pd
import numpy as np

# plot_eeg()

df = load_arff('filtered_output.arff')

X = df.iloc[:, :-1].to_numpy()
y = df.iloc[:, -1].to_numpy(dtype=int)

bs = 8

#One-Hot Enconding y
Y = (np.eye(2)[y])[:-(y.shape[0] % bs)]


#Standardization of X
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)

X_standardized = ((X - mean) / std)[:-(X.shape[0] % bs)]

model = LSTM(hidden_units=16, batch_size=bs)

model.fit(X_standardized, Y)

model.train(lr=0.001, interations=500)

model.predict(Xzao=X[:800], Yzao=Y[:800])