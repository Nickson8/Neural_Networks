import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
total = 400
generations = 20

def generate_data():
    samples = total

    cluster1 = np.random.randn(samples, 2) + np.array([10,10])
    cluster2 = np.random.randn(samples, 2) + np.array([-10,-10])
    cluster3 = np.random.randn(samples, 2) + np.array([-10,10])
    cluster4 = np.random.randn(samples, 2) + np.array([10,-10])

    X = np.vstack([cluster1, cluster2, cluster3, cluster4])

    y = np.hstack([np.zeros(samples), np.zeros(samples), np.ones(samples), np.ones(samples)])

    # Shuffle the dataset
    indices = np.random.permutation(len(X))  # Generate shuffled indices
    X, y = X[indices], y[indices]  # Apply shuffled indices to X and y

    return X, y

# Function to plot lines with weights from the matrix
def plot_lines(W, x_range):
    for i in range(W.shape[1]):  # Iterate over columns (each line equation)
        xw, yw, b = W[:, i]  # Extract slope and intercept
        y_range = -xw/yw * x_range - b/yw
        plt.plot(x_range, y_range, label=f"Line {i+1}")

def show_data(data_T, boundry):
    X, y = data_T

    plt.subplot(1, 2, 1)

    plt.scatter(X[y == 0, 0], X[y == 0, 1], c='blue', label='Class 0', s=10)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='red', label='Class 1', s=10)


    # Plot the lines after showing points
    plot_lines(boundry[0], X[:, 0])

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.ylim((-35,35))

    plt.legend()
    plt.grid(True)

    # ---- Step 2: Transform Points to New Coordinate System ----
    # Extract only the first two rows (ignoring bias) to form a transformation matrix
    W_transform = boundry[0][:2, :]  # Shape (2, N_lines)

    # Transform the points into the new system
    transformed_points = np.linalg.inv(W_transform.T) @ X.T  # Apply inverse transformation
    transformed_points = transformed_points.T  # Reshape back to (N_points, 2)

    # Extract new x and y values
    new_x_vals, new_y_vals = transformed_points[:, 0], np.tanh(transformed_points[:, 1])

    # ---- Step 3: Plot Transformed Points ----
    plt.subplot(1, 2, 2)
    plt.scatter(transformed_points[y == 0, 0], np.tanh(transformed_points[y == 0, 1]), c='blue', label='Class 0', s=20)
    plt.scatter(transformed_points[y == 1, 0], np.tanh(transformed_points[y == 1, 1]), c='red', label='Class 1', s=20)

    # Plot the lines after showing points
    plot_lines(boundry[1], transformed_points[:, 0])

    # Draw new coordinate axes (basis vectors)
    plt.axhline(0, color="black", linewidth=0.5)
    plt.axvline(0, color="black", linewidth=0.5)
    plt.grid(True, linestyle="--")

    plt.legend()
    plt.xlabel("New X-axis")
    plt.ylabel("New Y-axis")
    plt.ylim((-2,4))
    plt.title("Transformed Coordinate System")

    plt.show()


class NeuralNet():
    def __init__(self, layers_sizes, hidden_af=['tanh'], output_af=['lin']):
        self.depth = layers_sizes
        self.layers_sizes = layers_sizes
        self.layers_sizes.append(1)
        self.W = []
    
    def fit(self, X):
        self.X = X
        input_size = X.shape[1]
        Wl = [[] for _ in range(self.depth)]


        for layer in range(self.depth):
            for percptron in range(self.layers_sizes[layer]):
                if(layer == 0):
                    Wl[layer].append(np.random.randn(input_size + 1)) #add 1 for the bias
                else:
                    Wl[layer].append(np.random.randn(self.layers_sizes[layer-1] + 1))
        
        W = []
        for layer in Wl:
            W.append(np.array(layer).T)
        
        self.W = W
        
        return W
    
    def cross_entropy_derivative(self, w, X, y):
        Z = (X @ w).reshape(1,-1)
        Z = Z[0]

        yh = 1 / (1+ np.exp(-Z))

        return (yh - y)
    
    def predict(self, X):
        X = np.hstack((X, np.ones((X.shape[0], 1))))
        responses = X

        for index, layer in enumerate(self.W):
            if (index == len(self.W)-1):
                responses = responses @ layer
            else:
                responses = (np.hstack((np.tanh(responses @ layer), np.ones((responses.shape[0], 1)))))
        
        return (responses >= 1).astype(int)
    
    def accuracy(self, target, prediction):
        print(f"Target:")
        for v in target[:40]:
            print(int(v), end=" ")
        
        print(f"\nPredction:")
        for v in prediction[:40]:
            print(v[0], end=" ")

        print(f"\n\nAccuracy: {np.mean(target.astype(int) == prediction.reshape(1,-1)[0]) * 100}%")
    
    def get_responses(self, X):
        X = np.hstack((X, np.ones((X.shape[0], 1))))
        responses = [X]

        for index, layer in enumerate(self.W):
            if (index == len(self.W)-1):
                break
            else:
                responses.append(np.hstack((np.tanh(np.matmul(responses[index], layer)), np.ones((responses[index].shape[0], 1)))))
            
        return responses
    
    def batch_generator(self, X, y, batch_size=30):
        for i in range(0, len(X), batch_size):
            yield X[i:i+batch_size], y[i:i+batch_size]

    def train(self, y, lr, interations=100):

        for _ in range(interations):
            for X, y in self.batch_generator(self.X, y, 4):

                #pegando os inputs que cada camada recebeu
                responses = self.get_responses(X)[::-1]

                #invertendo a lista de camadas
                self.W = self.W[::-1]

                back = self.W

                mem = self.cross_entropy_derivative(self.W[0], responses[0], y)


                #updating the last layer
                self.W[0] = self.W[0] - lr* np.mean((responses[0] * mem.reshape(-1,1)), axis=0).reshape(-1,1)


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

        return (self.W)



#==*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=**==*=*=*=*=*=*=*=*=*=*=*=**=

data = generate_data()
X,y = data

NN = NeuralNet(layers_sizes=[4,2], hidden_af='tanh', output_af='lin')

gp = 100

W = NN.fit(X[:total-gp])

boundry = NN.train(lr=0.1, interations=generations, y=y[:total-gp])

print(f"Result for {total} samples and {generations} generations:")
NN.accuracy(y[total-gp:], NN.predict(X[total-gp:]))


#show_data((X, y), boundry)