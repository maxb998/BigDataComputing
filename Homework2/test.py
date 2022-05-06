import sys
import os
import numpy as np

def main():
    
    p: np.ndarray = np.concatenate((np.arange(10), np.arange(10)))
    np.random.shuffle(p)

    print(p)

    p_sorted_ids: np.ndarray = p.argsort()

    print(p[p_sorted_ids])

    print(p_sorted_ids)

    print(p.shape)
    print(p_sorted_ids.shape)






if __name__ == "__main__":
    main()