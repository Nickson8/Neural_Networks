import numpy as np
from time import sleep

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
        self.vs = np.random.randn(self.bs, self.h)
        self.vs1 = self.vs

        #Memory cell Matrix (B x H) and Weights (N+1 x H) and (H+1 x H)
        self.mc = np.random.randn(self.bs, self.h)
        self.mc1 = self.mc
        self.mcx = np.random.randn(self.n+1, self.h)
        self.mch = np.random.randn(self.h+1, self.h)

        #Target Vector
        self.y = y

        #Forget Gate Weights
        self.fgx = np.random.randn(self.n+1, self.h)
        self.fgh = np.random.randn(self.h+1, self.h)
        #Input Gate Weights
        self.igx = np.random.randn(self.n+1, self.h)
        self.igh = np.random.randn(self.h+1, self.h)
        #Output Gate Weights
        self.ogx = np.random.randn(self.n+1, self.h)
        self.ogh = np.random.randn(self.h+1, self.h)

        #Predictor Weight Matrix
        self.pw = np.random.randn(self.h, self.y.shape[1])
    
    
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

    # def sigmoid(self, x):
    #     exp = np.exp(x)
    #     return np.where(
    #         x >= 0,
    #         1 / (1 + np.exp(-x)),  
    #         exp / (1 + exp)
    #     )
    
    def softmax(self, R):
        ps = [R[i, :] for i in range(R.shape[0])]
        ps = [np.exp(v) * (1/np.sum(np.exp(v))) for v in ps]
        
        return np.vstack(ps)

    def foward(self, X):
        #Fix for the offset
        vs = np.hstack((self.vs, np.ones((self.vs.shape[0], 1))))

        self.fg = self.sigmoid( (X @ self.fgx) + (vs @ self.fgh)  )
        self.ig = self.sigmoid( (X @ self.igx) + (vs @ self.igh)  )
        self.og = self.sigmoid( (X @ self.ogx) + (vs @ self.ogh)  )


        self.mc1 = self.mc
        self.gt = np.tanh( (X @ self.mcx) + (vs @ self.mch) )
        self.mc = (self.fg * self.mc) + ( self.ig * self.gt )

        self.vs1 = np.hstack((self.vs, np.ones((self.vs.shape[0], 1))))
        self.vs = self.og * np.tanh(self.mc)

        self.p = self.softmax(self.vs @ self.pw)

        return self.p
    
    def predict(self, Xzao, Yzao):
        mc = self.mc
        vs = self.vs
        Xzao2 = np.hstack((Xzao, np.ones((Xzao.shape[0], 1))))

        print("  P Y")

        for X, y in self.batch_generator(X=Xzao2, y=Yzao, batch_size=self.bs):
            #Fix for the offset
            vs = np.hstack((vs, np.ones((vs.shape[0], 1))))

            fg = self.sigmoid( (X @ self.fgx) + (vs @ self.fgh)  )
            ig = self.sigmoid( (X @ self.igx) + (vs @ self.igh)  )
            og = self.sigmoid( (X @ self.ogx) + (vs @ self.ogh)  )

            gt = np.tanh( (X @ self.mcx) + (vs @ self.mch) )
            mc = (fg * mc) + ( ig * gt )

            vs = og * np.tanh(mc)

            p = self.softmax(self.vs @ self.pw)

            print( np.hstack( (np.argmax(p, axis=1).reshape(-1,1), np.argmax(y, axis=1).reshape(-1,1)) ) )

    
    def batch_generator(self, X, y, batch_size):
        for i in range(0, len(X), batch_size):
            yield X[i:i+batch_size], y[i:i+batch_size]
    
    def clip_gradient_by_norm(self, grad, max_norm=5):
        """
        Clips the gradient to ensure its L2 norm does not exceed max_norm.

        Parameters:
        - grad (numpy.ndarray): The gradient matrix.
        - max_norm (float): The maximum allowed norm.

        Returns:
        - numpy.ndarray: The clipped gradient.
        """
        norm = np.linalg.norm(grad)  # Compute L2 norm
        if norm > max_norm:
            grad = grad * (max_norm / norm)  # Scale down
        return grad

    def train(self, lr, interations=100):
        print(f"Training with {interations} interations:")
        
        for _ in range(interations):
            #LR nn funciona, clipping tmb nn ta
            loss = np.array([])
            if(_ == 30):
                print("H:")
                memm = ((self.p - y) @ self.pw.T) * (self.og * (1 - (np.tanh(self.mc))**2 ))
                hh = -self.clip_gradient_by_norm(self.vs1.T @ (memm * self.mc1))
                xx = -self.clip_gradient_by_norm(X.T @ (memm * self.mc1))
                print(hh)
                print("X:")
                print(xx)
            if(_ == 50):
                print("H:")
                memm = ((self.p - y) @ self.pw.T) * (self.og * (1 - (np.tanh(self.mc))**2 ))
                hh = -self.clip_gradient_by_norm(self.vs1.T @ (memm * self.mc1))
                xx = -self.clip_gradient_by_norm(X.T @ (memm * self.mc1))
                print(hh)
                print("X:")
                print(xx)
            
            # if(_ == 1):
            #     print("Fg:")
            #     print(self.fg)
            # if(_ == 10):
            #     print("Fg")
            #     print(self.fg)
            for X, y in self.batch_generator(self.X, self.y, self.bs):

                #Updating the prediction layer
                self.foward(X=X)
                self.pw = self.pw - lr * self.clip_gradient_by_norm(self.vs.T @ (self.p - y))

                #Loss
                losses0 = -(y * np.log(self.p))
                losses = np.hstack([row[row != 0] for row in losses0])
                loss = np.hstack((loss, losses))

                # print("\n=*=*=*=*=*=*=*=*=*=*=*")
                # print(((self.p - y) @ self.pw.T))
                # print((self.og * (1 - (np.tanh(self.mc))**2 )))
                # print("\n=*=*=*=*=*=*=*=*=*=*=*")

                mem = ((self.p - y) @ self.pw.T) * (self.og * (1 - (np.tanh(self.mc))**2 ))

                #Updating the memory cell weights
                self.mch = self.mch - lr * self.clip_gradient_by_norm(self.vs1.T @ (mem * self.ig))
                self.mcx = self.mcx - lr * self.clip_gradient_by_norm(X.T @ (mem * self.ig))

                #Updating the forget gate weights
                self.fgh = self.fgh - lr * self.clip_gradient_by_norm(self.vs1.T @ (mem * self.mc1))
                self.fgx = self.fgx - lr * self.clip_gradient_by_norm(X.T @ (mem * self.mc1))

                #Updating the input gate weights
                self.igh = self.igh - lr * self.clip_gradient_by_norm(self.vs1.T @ (mem * self.gt))
                self.igx = self.igx - lr * self.clip_gradient_by_norm(X.T @ (mem * self.gt))

                mem = (self.p - y) @ self.pw.T * np.tanh(self.mc)

                #Updating the output gate weights
                self.ogh = self.ogh - lr * self.clip_gradient_by_norm(self.vs1.T @ mem)
                self.ogx = self.ogx - lr * self.clip_gradient_by_norm(X.T @ mem)
            print(f"Loss of interation {_+1}: {np.mean(loss)}")