import numpy as np

def uniform_crossover(x1, x2):
    alpha = np.random.randint(0, 2, size=x1.shape)
    y1 = alpha * x1 + (1 - alpha) * x2
    y2 = alpha * x2 + (1 - alpha) * x1
    return y1, y2
