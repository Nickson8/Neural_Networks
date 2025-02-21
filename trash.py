import numpy as np

X = np.array([[1, 2, 3, 4], 
            [5, 6, 7, 8], 
            [9, 10, 11, 12]])
ar = np.array([[2, 1, 7, 5, 3]])

ar2 = np.array([[-2, 1, 0],
                [2, -9, 1],
                [-8, 13, 4]])

ar3 = np.array([[0, 1, 0, 2, 1, 0],
                [0, 1, 0, 3, 9, 0],
                [0, 1, 0, 4, 1, 0],
                [0, 1, 0, 2, 1, 0],
                [0, 1, 0, 3, 9, 0],
                [0, 1, 0, 4, 1, 0]])
i = 0
for __ in range(ar3.shape[0]//2):
    j=0
    for __ in range(ar3.shape[1]//2):
        print(f"{i},{j}")
        print(ar3[i:i+2, j:j+2])
        j += 2
    i += 2