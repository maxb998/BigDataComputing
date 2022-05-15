import time
import sys
import math
import os
import numpy as np

def readVectorsSeq(filename):
    with open(filename) as f:
        result = [tuple(map(float, i.split(','))) for i in f]
    return result
    
    
def euclidean(point1,point2):
    dist = np.sum(np.square(point1 - point2),axis=1)
    return np.sqrt(dist)

def Bz(x,r,Z):
    X = np.array(x)
    idx = np.where(euclidean(X,Z) <= r)   
    return idx

def SeqWeightedOutliers(P,W,k,z,alpha):
    # find r which is the minimum half distance between the first k+z+1 points
    P_kz1 = np.array(P[:k+z+1]) # We take only the first k+z+1 points from the pointset P
    r_min = np.zeros(shape=(k+z+1,k+z+1))
    for i in range(k+z+1):
        r_min[i] = euclidean(P_kz1,P_kz1[i])
        r_min[i,i] = np.max(r_min[i]) # We assign the maximum value of the row to avoid considering the diagonal(which is all zeros)

    r = np.min(r_min)/2
    
    # r_i is the initial guess
    r_i = r
    # n_guess is the number of cycles
    n_guess = 0
    w = np.array(W)
    while(True):
        Z = P.copy()
        S = []
        Wz = sum(W)
        n_guess += 1
        
        while ((len(S)<k) and (Wz>0)):
            
            max_v = 0
            for x in Z:
                idx = Bz(x,(1+2*alpha)*r, Z) # Indices of the points inside the ball
                ball_weight = np.sum(w[idx])
                if ball_weight > max_v:
                    max_v = ball_weight
                    newcenter = x
            S.append(newcenter) # Add the new center found
            idx = Bz(newcenter,(3+4*alpha)*r, Z)
            Z = np.delete(Z,idx,0)
            Wz = Wz - np.sum(w[idx])
        if Wz<=z:
            # r_f is the final guess
            r_f = r
            return S, r_i, r_f, n_guess
        else:
            r = 2*r
            
                
def ComputeObjective(P,S,z):

    min_dist_from_centers = np.zeros(len(P))
    dist_from_centers = np.zeros(shape=(len(P),len(S)))
    points = np.array(P)
    centers = np.array(S)

    for i in range(len(P)):
        dist_from_centers[i] = euclidean(points[i],centers)

    min_dist_from_centers = np.min(dist_from_centers, axis=1)
    result = np.sort(min_dist_from_centers, axis=None)
    return result[-(z+1)]

def main():
    
    # Check argv lenght and content
    assert len(sys.argv) == 4, "Usage: G001HW2.py <Filename> <k> <z>"
    
    filename = sys.argv[1]
    assert os.path.isfile(filename), "File or folder not found"
    # Read the points in the input file into a list of tuple
    inputPoints = readVectorsSeq(filename)

    # Create a list of ones 'weights' of the same cardinality of inputPoints
    n = len(inputPoints)
    weights = [1]*n
    # Read the number of centers
    k = sys.argv[2]
    assert k.isdigit, "K must be an integer value"
    k = int(k)
    assert k >= 0, "K must be positive"
    # Read the number of allowed outliers
    z = sys.argv[3]
    assert z.isdigit, "Z must be an integer value"
    z = int(z)
    assert z >= 0, "Z must be positive"
    # Calculating the time (in milliseconds) required by the execution of SeqWeightedOutliers
    start = time.time() * 1000
    # Method that implements the weighted variant of kcenterOUT to return the set of centers S 
    solution, r_i, r_f, n_guess = SeqWeightedOutliers(inputPoints,weights,k,z,0)
    end = time.time() * 1000
    # Method that returns the largest distange among all distances(x,S), for x in P, excluding the first z largest disatnces
    objective =  ComputeObjective(inputPoints,solution,z)

    # Output
    print("Input size n =", n)
    print("Number of centers k =", k)
    print("Number of outliers z =", z)
    print("Initial guess =", r_i)
    print("Final guess =", r_f)
    print("Number of guesses =", n_guess)
    print("Objective function =", objective)
    print("Time of SeqWeightedOutliers =", end-start)


if __name__ == "__main__":
    main()
