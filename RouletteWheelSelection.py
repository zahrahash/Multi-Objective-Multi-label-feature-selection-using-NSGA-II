import numpy as np

def roulette_wheel_selection(P):
    r = np.random.rand()
    c = np.cumsum(P)
    i = np.where(r <= c)[0][0]
    return i
