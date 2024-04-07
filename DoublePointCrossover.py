import numpy as np

def double_point_crossover(x1, x2):
    n = len(x1)
    c1, c2 = sorted(np.random.choice(range(1, n), 2, replace=False))
    y1 = np.concatenate([x1[:c1], x2[c1:c2], x1[c2:]])
    y2 = np.concatenate([x2[:c1], x1[c1:c2], x2[c2:]])
    return y1, y2
