import sys
import os
import csv
import time
import math

def readVectorsSeq(filename):
    with open(filename) as f:
        result = [tuple(map(float, i.split(','))) for i in f]
    return result

    
def euclidean(point1,point2):
    res = 0
    for i in range(len(point1)):
        diff = (point1[i]-point2[i])
        res +=  diff*diff
    return math.sqrt(res)


def SeqWeightedOutliers(P,W,k,z,alpha):

    # find r which is the minimum half distance between the first k+z+1 points
    r = euclidean(P[0], P[1])
    for i in range(k+z):
        for j in range(i+1,k+z+1):
            eucl_dist = euclidean(P[i], P[j])
            if eucl_dist < r:
                r = eucl_dist
    r = r / 2

    while True:
        a = 10


    return 0


def main():

    # Check argv lenght and content
    assert len(sys.argv) == 4, "Usage: <Filename> <K> <Z>"

    filename = sys.argv[1]
    assert os.path.isfile(filename), "File or folder not found"
    inputPoints = readVectorsSeq(filename=filename)
    
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