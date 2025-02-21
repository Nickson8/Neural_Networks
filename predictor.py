import numpy as np

class Predictor():
    def __init__(self, layers_sizes, hidden_units,hidden_af=['tanh'], output_af=['softmax']):
        self.depth = len(layers_sizes)+1
        self.layers_sizes = layers_sizes
        self.layers_sizes.append(2)
        self.W = []

        Wl = [[] for _ in range(self.depth)]



        for layer in range(self.depth):
            for percptron in range(self.layers_sizes[layer]):
                if(layer == 0):
                    Wl[layer].append(np.random.randn(hidden_units + 1)) #add 1 for the bias
                else:
                    Wl[layer].append(np.random.randn(self.layers_sizes[layer-1] + 1))
        
        W = []
        for layer in Wl:
            W.append(np.array(layer).T)
        
        self.W = W
        
    
    def cross_entropy_derivative(self, W, X, Y):
        Z = X @ W
        P = self.softmax(Z)

        return (P - Y)
    
    def softmax(self, R):
        ps = [R[i, :] for i in range(R.shape[0])]
        ps = [np.exp(v) * (1/np.sum(np.exp(v))) for v in ps]
        
        return np.vstack(ps)

    def predict(self, X):
        X = np.hstack((X, np.ones((X.shape[0], 1))))
        responses = X

        for index, layer in enumerate(self.W):
            if (index == len(self.W)-1):
                responses = self.softmax(responses @ layer)
            else:
                responses = (np.hstack((np.tanh(responses @ layer), np.ones((responses.shape[0], 1)))))
        
        return (responses >= 1).astype(int)

    
    def get_responses(self, X):
        X = np.hstack((X, np.ones((X.shape[0], 1))))
        responses = [X]

        for index, layer in enumerate(self.W):
            if (index == len(self.W)-1):
                break
            else:
                responses.append(np.hstack((np.tanh(np.matmul(responses[index], layer)), np.ones((responses[index].shape[0], 1)))))
            
        return responses

    def train(self, X, Y, lr):

        #pegando os inputs que cada camada recebeu
        responses = self.get_responses(X)[::-1]

        #invertendo a lista de camadas
        self.W = self.W[::-1]

        back = self.W

        mem = self.cross_entropy_derivative(self.W[0], responses[0], Y)


        #updating the last layer
        columns_ll = [self.W[0][:, i].reshape(-1,1) for i in range(self.W[0].shape[1])] #list of the array weights of each perceptron
        for iw, w in enumerate(columns_ll):
            self.W[0][:, iw] = self.W[0][:, iw] - lr* np.mean((responses[0] * mem.reshape(-1,1)), axis=0).reshape(-1,1)


        #updating the second final layer
        columns_sl = [self.W[1][:, i].reshape(-1,1) for i in range(self.W[1].shape[1])] #list of the array weights of each perceptron

        for iw, w in enumerate(columns_sl):
            self.W[1][:, iw] = self.W[1][:, iw] - lr * np.mean(((responses[1] * (1 - (np.tanh(responses[1] @ w)**2)) ) *back[0][:, 0][iw]) * mem.reshape(-1,1), axis=0)
        #updating the mem
        mem = mem * self.W[0][0][0]
        

        #updating the rest
        for il, layer in enumerate(self.W[self.layers_sizes[-2]+1:]):
            columns = [layer[:, i] for i in range(layer.shape[1])] #list of the array weights of each perceptron
            nIl = il+self.layers_sizes[-2]+1
            for iw, w in enumerate(columns):
                self.W[nIl][:, iw] = (

self.W[nIl][:, iw] - lr * np.mean(( ( (responses[nIl] * (1 - (np.tanh(responses[nIl] @ w)**2)) ) * ((1 - (responses[nIl-1][0]**2)) * back[nIl-1][:, 0][iw]) ) * mem.reshape(-1,1) ), axis=0))
                
            #updating the mem
            mem = mem * ((1 - (responses[nIl-1][0]**2)) * back[nIl-1][:, 0][0])

        #revertendo a lista de camadas
        self.W = self.W[::-1]

        return (mem)