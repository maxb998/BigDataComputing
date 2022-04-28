import sys
import os
import csv
import numpy as np

def main():

    # Check argv lenght and content
    assert len(sys.argv) == 4, "Usage: <Filename> <K> <Z>"

    filename = sys.argv[1]
    assert os.path.isfile(filename), "File or folder not found"
    try:
        f = open(filename)
        data = np.loadtxt(fname=filename, dtype=np.float32, delimiter=',')
    finally:
        f.close()
    
    K = sys.argv[2]
    assert K.isdigit, "K must be an integer value"
    K = int(K)
    assert K >= 2, "K must be at least 2"

    Z = sys.argv[3]
    assert Z.isdigit, "Z must be an integer value"
    Z = int(Z)
    assert Z >= 0, "K must be positive"

    
    
    






if __name__ == "__main__":
    main()