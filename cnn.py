import numpy as np

np.random.seed(42)

class CNN:
    def __init__(self, kernels_sizes):
        self.kernels_sizes = kernels_sizes
        #First Layer Kernels
        self.l1_kernels = np.random.randn(kernels_sizes[0],3,3)

        #Second Layer Kernels
        self.l2_kernels = np.random.randn(kernels_sizes[1],3,3)

        #Third Layer Kernels
        self.l3_kernels = np.random.randn(kernels_sizes[2],3,3)

        #Prediction Layer Weights
        n = kernels_sizes[0] * kernels_sizes[1] * kernels_sizes[2]
        self.pw = np.random.randn(n,10)
    
    def fit(self, Imgs, Y):
        self.Imgs = Imgs
        self.Y = Y

    def ReLu(self, X):
        return max(0, X)
    
    def conv_(self, img_part, kernel):
        return self.ReLu(np.sum(img_part * kernel))
    
    def Conv(self, img, kernel):
        #Convolution with 3x3 kernel and stride 1
        new = np.empty((img.shape[0] -2, img.shape[1] -2))
        for i in range(img.shape[0] -2):
            for j in range(img.shape[1] -2):
                new[i][j] = self.conv_(img_part=img[i:i+3, j:j+3], kernel=kernel)
        
        return new
    
    def pool_(self, img_part):
        #2x2 max-pooling
        return np.max(img_part)
    
    def Pool(self, img):
        #2x2 max-pooling with stride 2
        new = np.empty((img.shape[0]//2, img.shape[1]//2))
        i=0
        for l in range(img.shape[0]//2):
            j=0
            for c in range(img.shape[1]//2):
                new[l][c] = self.pool_(img_part=img[i:i+2, j:j+2])
                j+=2
            i+=2
        
        return new
    
    def softmax(self, R):
        s = np.sum(np.exp(R))

        return np.exp(R) * 1/s

    def foward(self, img):
        #First Layer
        Za = []
        A = []
        Ra = []
        for kernel in range(self.l1_kernels.shape[0]):
            Ra.append( self.Pool( self.Conv(img=img, kernel=self.l1_kernels[kernel]) ) )
        
        #Second Layer
        l2_results = []
        for feature_map in Ra:
            for kernel in range(self.l2_kernels.shape[0]):
                l2_results.append( self.Pool( self.Conv(img=feature_map, kernel=self.l2_kernels[kernel]) ) )
        
        #Third Layer
        l3_results = []
        for feature_map in l2_results:
            for kernel in range(self.l3_kernels.shape[0]):
                l3_results.append( self.Pool( self.Conv(img=feature_map, kernel=self.l3_kernels[kernel]) ) )
        
        R = np.array([i[0] for i in l3_results]).reshape(1,-1)[0]

        #Prediction layer
        Z = R @ self.pw
        P = self.softmax( Z )

        return (R, P)
    
    def cross_entropy_loss(self, Y, P):
        return np.max(-Y * np.log(P))
    
    def train(self, interations, lr):
        for epoch in interations:
            loss = 0
            for Img, Y in zip(self.Imgs, self.Y):
                #R é o input da camada Softmax (antes de ser mult. pelos pesos) e P é o output da Softmax
                R, P = self.foward(img=Img)
                loss += self.cross_entropy_loss(Y, P)

                t = np.argmax(Y)

                mem = 1

                pw = self.pw

                #Updating Prediction Layer
                for c in range(self.pw.shape[1]):
                    Pc = P[c]
                    if (t == c):
                        # (d Loss / d Pt) * (d Pt / d Zt)
                        mem *= (Pc-1)

                        for l in range(self.pw.shape[0]):   
                            self.pw[l][c] -= lr * (Pc-1) * R[l]
                    else:
                        for l in range(self.pw.shape[0]):   
                            self.pw[l][c] -= lr * Pc * R[l]
                
                
                #Updating the Third Layer Kernels

            print(loss/len(self.Imgs))



#=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*==*=*=*=*=*=*=*=*=*=*=*==*=*=*=*=*=*=*=*=*=*=*==*=*=*=*=*=*=*=*=*=*=*=

cnn = CNN([5,2,2])

img = np.random.randn(28,28)

print( cnn.cross_entropy_loss( np.array([0,0,0,0,1,0,0,0,0,0]) , cnn.foward(img) ) )