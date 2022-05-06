import time
import sys
import math
import os


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

def Bz(x,r,Z):
    result = []
    for y in Z:
        if euclidean(x,y)<=r:
            result.append(y)
    return result

def SeqWeightedOutliers(P,W,k,z,alpha):
    # find r which is the minimum half distance between the first k+z+1 points
    r_min = []
    for i in range(k+z+1):
        for j in range(k+z+1):
            if i!=j:
                r_min.append(euclidean(P[i], P[j]))

    r = min(r_min)/2
    # r_i is the initial guess
    r_i = r
    # n_guess is the number of cycles
    n_guess = 0
    while(True):
        Z = P.copy()
        S = []
        Wz = sum(W)
        n_guess += 1
        temp = 0
        while ((len(S)<k) and (Wz>0)):
            max_v = 0
            for x in P:
                for j in Bz(x,(1+2*alpha)*r, Z):
                    temp += W[P.index(j)] 
                ball_weight = temp 
                if ball_weight > max_v:
                    max_v = ball_weight
                    newcenter = x
            S.append(x)
            for y in Bz(newcenter,(3+4*alpha)*r, Z):
                Z.remove(y)
                Wz = Wz - W[P.index(y)]
        if Wz<=z:
            # r_f is the final guess
            r_f = r
            return S, r_i, r_f, n_guess
        else:
            r = 2*r
            
                
def ComputeObjective(P,S,z):
    result = []
    for i in P:
        for j in S:
            result.append(euclidean(i,j))

    result.sort()
    for i in range(z):
        result.pop()
    return result[-1]

def main():

    # Check argv lenght and content
    assert len(sys.argv) == 4, "Usage: python G001HW2.py <Filename> <k> <z>"
    
    filename = sys.argv[1]
    assert os.path.isfile(filename), "File or folder not found"
    # Read the points in the input file into a list of tuple
    inputPoints = readVectorsSeq(filename)
    # Create a list of ones 'weights' of the same cardinality of inputPoints
    n = len(inputPoints)
    weights = [1]*n
    
    k = sys.argv[2]
    assert k.isdigit, "K must be an integer value"
    k = int(k)
    assert k >= 0, "K must be positive"
    
    z = sys.argv[3]
    assert z.isdigit, "Z must be an integer value"
    z = int(z)
    assert z >= 0, "Z must be positive"
    
    solution, r_i, r_f, n_guess = SeqWeightedOutliers(inputPoints,weights,k,z,0)
    objective =  ComputeObjective(inputPoints,solution,z)

    # Output
    print("Input size n =", n)
    print("Number of centers k =", k)
    print("Number of outliers z =", z)
    print("Initial guess =", r_i)
    print("Final guess =", r_f)
    print("Number of guesses =", n_guess)
    print("Objective function =", objective)
    #print("Time of SeqWeightedOutliers =", )
     
            



if __name__ == "__main__":
    main()
