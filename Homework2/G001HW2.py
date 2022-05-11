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
        
        while ((len(S)<k) and (Wz>0)):
            
            max_v = 0
            for x in Z:
                temp = 0
                for j in Bz(x,(1+2*alpha)*r, Z):
                    temp += W[P.index(j)] 
                ball_weight = temp 
                if ball_weight > max_v:
                    max_v = ball_weight
                    newcenter = x
            S.append(newcenter)
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

    min_dist_from_centers = []
    dist_from_centers = [0.] * len(S)
    
    for i in range(len(P)):
        for j in range(len(S)):
            dist_from_centers[j] = euclidean(P[i],S[j])
        min_dist_from_centers.append(min(dist_from_centers)) # Adding the minimum distance of the single point x from the all the centers
        min_dist_from_centers.sort() # Sorting at every cycle to reduce the computational time of the last sort since almost all element will be already sorted
    min_dist_from_centers.sort()
    return min_dist_from_centers[-(z+1)]

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
