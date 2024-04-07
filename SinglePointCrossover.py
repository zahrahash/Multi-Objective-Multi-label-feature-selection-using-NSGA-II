import numpy as np

def single_point_crossover(x1, x2):
    n = len(x1)
    c = np.random.randint(1, n)
    y1 = np.concatenate([x1[:c], x2[c:]])
    y2 = np.concatenate([x2[:c], x1[c:]])
    return y1, y2
