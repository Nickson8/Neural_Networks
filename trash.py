import numpy as np

def softmax(R):
    ps = [R[i, :] for i in range(R.shape[0])]
    ps = [np.exp(v) * (1/np.sum(np.exp(v))) for v in ps]
    
    return np.mat(ps)


X = np.array([[1, 2, 3, 4], 
            [5, 6, 7, 8], 
            [9, 10, 11, 12]])
ar = np.array([[2, 1],
               [3, 4],
               [4, 7]])

X2 = np.hstack( ( X, np.ones(X.shape[0]).reshape(-1,1) ) )

ar2 = np.random.randn(3,2)

print(ar2)

