import numpy as np

def sigmoid(x):
    return np.where(
        x >= 0,
        1 / (1 + np.exp(-x)),  
        np.exp(x) / (1 + np.exp(x))  
    )

def clip_gradient_by_norm(grad, max_norm=5):
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

X = np.array([[1, 2, 3, 4], 
            [5, 6, 7, 8], 
            [9, 10, 11, 12]])
ar = np.array([[2, 1, 7, 5, 3]])

ar2 = np.array([[-300, 1],
                [20, -9],
                [-80, 40]])

print(clip_gradient_by_norm(ar2))
print(0.01 * clip_gradient_by_norm(ar2))

