import numpy as np

class LSTM():
    def __init__(self, hidden_units, batch_size, hidden_af=['tanh']):
        self.bs = batch_size
        self.h = hidden_units
        self.W = []
    
    def fit(self, X, y):
        #Input Size
        self.n = X.shape[1]

        #Input Matrix (B x N+1)
        self.X = np.hstack((X, np.ones((X.shape[0], 1))))

        #Visible State Matrix (B x H)
        self.vs = np.random.randn((self.bs, self.h))
        self.vs1 = self.vs

        #Memory cell Matrix (B x H) and Weights (N+1 x H) and (H+1 x H)
        self.mc = np.random.randn((self.bs, self.h))
        self.mc1 = self.mc
        self.mcx = np.random.randn((self.n+1, self.h))
        self.mch = np.random.randn((self.h+1, self.h))

        #Target Vector
        self.y = y

        #Forget Gate Weights
        self.fgx = np.random.randn((self.n+1, self.h))
        self.fgh = np.random.randn((self.h+1, self.h))
        #Input Gate Weights
        self.igx = np.random.randn((self.n+1, self.h))
        self.igh = np.random.randn((self.h+1, self.h))
        #Output Gate Weights
        self.ogx = np.random.randn((self.n+1, self.h))
        self.ogh = np.random.randn((self.h+1, self.h))

        #Predictor Weight Matrix
        self.pw = np.random.randn((self.h, 1))
    
    
    def accuracy(self, target, prediction):
        print(f"Target:")
        for v in target[:40]:
            print(int(v), end=" ")
        
        print(f"\nPredction:")
        for v in prediction[:40]:
            print(v[0], end=" ")

        print(f"\n\nAccuracy: {np.mean(target.astype(int) == prediction.reshape(1,-1)[0]) * 100}%")
    

    def sigmoid(self, X):
        return 1/(1 + np.exp(-X))
    
    def softmax(self, R):
        ps = [R[i, :] for i in range(R.shape[0])]
        ps = [np.exp(v) * np.sum(np.exp(v)) for v in ps]
        
        return np.mat(ps)

    def foward(self, X):
        #Fix for the offset
        vs = np.hstack((self.vs, np.ones((self.vs.shape[0], 1))))

        self.fg = self.sigmoid( (self.fgx @ X) + (self.fgh @ vs)  )
        self.ig = self.sigmoid( (self.igx @ X) + (self.igh @ vs)  )
        self.og = self.sigmoid( (self.ogx @ X) + (self.ogh @ vs)  )

        self.mc1 = self.mc
        self.gt = np.tanh( (self.mcx @ X) + (self.mch @ vs) )
        self.mc = (self.fg * self.mc) + ( self.ig * self.gt )

        self.vs1 = np.hstack((self.vs, np.ones((self.vs.shape[0], 1))))
        self.vs = self.og * np.tanh(self.mc)

        self.p = self.softmax(self.vs @ self.pw)

        return self.p

    
    def batch_generator(self, X, y, batch_size):
        for i in range(0, len(X), batch_size):
            yield X[i:i+batch_size], y[i:i+batch_size]

    def train(self, lr, interations=100):

        for _ in range(interations):
            for X, y in self.batch_generator(self.X, self.y, self.bs):

                #Updating the prediction layer
                self.foward(X=X)
                self.pw = self.pw - lr * self.vs.T @ (self.p - y)

                mem = (self.p - y) @ self.pw.T * self.og * (1 - (np.tanh(self.mc))**2 )

                #Updating the memory cell weights
                self.mch = self.mch - lr * self.vs1.T @ (mem * self.ig)
                self.mcx = self.mcx - lr * X.T @ (mem * self.ig)

                #Updating the forget gate weights
                self.fgh = self.fgh - lr * self.vs1.T @ (mem * self.mc1)
                self.fgx = self.fgx - lr * X.T @ (mem * self.mc1)

                #Updating the input gate weights
                self.igh = self.igh - lr * self.vs1.T @ (mem * self.gt)
                self.igx = self.igx - lr * X.T @ (mem * self.gt)

                mem = (self.p - y) @ self.pw.T * np.tanh(self.mc)

                #Updating the output gate weights
                self.ogh = self.ogh - lr * self.vs1.T @ mem
                self.ogx = self.ogx - lr * X.T @ mem