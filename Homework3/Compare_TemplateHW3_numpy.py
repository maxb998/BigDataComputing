# Import Packages
from pyspark import SparkConf, SparkContext
import numpy as np
import time
import random
import sys
import math

# needed fot typing purpuses (I think...)
from typing import List, Tuple, Iterable
from pyspark import RDD

# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# MAIN PROGRAM
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def main():
    # Checking number of cmd line parameters
    assert len(sys.argv) == 5, "Usage: python Homework3.py filepath k z L"

    # Initialize variables
    filename = sys.argv[1]
    k = int(sys.argv[2])
    z = int(sys.argv[3])
    L = int(sys.argv[4])

    # Set Spark Configuration
    conf = SparkConf().setAppName('MR k-center with outliers')
    sc = SparkContext(conf=conf)
    sc.setLogLevel("WARN")

    # Read points from file
    start = time.time()
    inputPoints = sc.textFile(filename, L).map(lambda x : strToVector(x)).repartition(L).cache()
    N = inputPoints.count()
    end = time.time()

    '''
    # Pring input parameters
    print("File :" + filename)
    print("Number of points N =", N)
    print("Number of centers k =", k)
    print("Number of outliers z =", z)
    print("Number of partitions L =", L)
    print("Time to read from file:", str((end-start)*1000.), "ms")
    '''

    # Solve the problem
    start = time.time()
    solution = MR_kCenterOutliers(inputPoints, k, z, L)
    end = time.time()
    print("Time to complete round 1:", str((end-start)*1000), " ms")
    
    # Compute the value of the objective function
    start = time.time()
    objective = computeObjective(inputPoints, solution, z)
    end = time.time()
    #print("Objective function =", objective)
    #print("Time to compute objective function:", str((end-start)*1000.), "ms")
    



# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# AUXILIARY METHODS
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method strToVector: input reading
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def strToVector(str: str) -> Tuple[float, ...]:
    out = tuple(map(float, str.split(',')))
    return out

# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method squaredEuclidean: squared euclidean distance, in case you need it
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def squaredEuclidean(point1: Tuple[float, ...], point2: Tuple[float, ...]) -> float:
    res = 0
    for i in range(len(point1)):
        diff = (point1[i]-point2[i])
        res +=  diff*diff
    return res

# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method euclidean:  euclidean distance, in case you need it
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def euclidean(point1: Tuple[float, ...], point2: Tuple[float, ...]) -> float:
    res = 0
    for i in range(len(point1)):
        diff = (point1[i]-point2[i])
        res +=  diff*diff
    return math.sqrt(res)

# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method MR_kCenterOutliers: MR algorithm for k-center with outliers
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def MR_kCenterOutliers(points: RDD, k: int, z: int, L: int) -> List[Tuple[float, ...]]:

    
    #------------- ROUND 1 ---------------------------

    coreset = points.mapPartitions(lambda iterator: extractCoreset(iterator, k+z+1))
    
    #------------- END OF ROUND 1 --------------------

    
    #------------- ROUND 2 ---------------------------
    elems = coreset.collect()
    coresetPoints = []
    coresetWeights = []
    for i in elems:
        coresetPoints.append(i[0])
        coresetWeights.append(i[1])
    
    leng = len(elems)
    print("First point in coreset:", coresetPoints[0], " W=", coresetWeights[0])
    print("Middle point in coreset:", coresetPoints[int(leng/2)], " W=", coresetWeights[int(leng/2)])
    print("Last point in coreset:", coresetPoints[leng-1], " W=", coresetWeights[leng-1])
    
    # ****** ADD YOUR CODE
    # ****** Compute the final solution (run SeqWeightedOutliers with alpha=2)
    # ****** Measure and print times taken by Round 1 and Round 2, separately
    # ****** Return the final solution
    S = SeqWeightedOutliers(coresetPoints, coresetWeights, k, z, 2.)
    return S
   

# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method extractCoreset: extract a coreset from a given iterator
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def extractCoreset(iter: Iterable[Tuple[float, ...]], points: int) -> List[Tuple]:
    temp = list(iter)   # convert to list before and than to ndarray
    partition = np.array(temp)
    centers = kCenterFFT(partition, points)
    weights = computeWeights(partition, centers)
    
    c_w = list()
    for i in range(centers.shape[0]):
        entry = (tuple(centers[i]), weights[i])
        c_w.append(entry)
    # return weighted coreset
    return c_w
    
    
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method kCenterFFT: Farthest-First Traversal
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def kCenterFFT(points: np.ndarray, k: int) -> np.ndarray:
    n = points.shape[0] # number of points in dataset
    random.seed(2022)
    idx_rnd = random.randint(0, n-1)
    centers = np.zeros(shape=(k, points.shape[1]), dtype=float)

    centers[0] = points[idx_rnd]
    dist_from_nearest_center = np.sum(a=np.square(points - points[idx_rnd]), axis=1, dtype=float)  #compute square distance between first center and all points (including itself)

    for i in range(k-1):
        new_center_idx = np.argmax(a=dist_from_nearest_center)
        centers[i+1] = points[new_center_idx]
        dist_from_new_center = np.sum(a=np.square(points - centers[i+1]), axis=1, dtype=float)
        np.minimum(dist_from_nearest_center, dist_from_new_center, out=dist_from_nearest_center)

    return centers

# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method computeWeights: compute weights of coreset points
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def computeWeights(points: np.ndarray, centers: np.ndarray) -> np.ndarray:
    weights = np.zeros(shape=centers.shape[0], dtype=int)

    distances = np.zeros(shape=centers.shape[0], dtype=float)
    for i in range(points.shape[0]):
        np.sum(a=np.square(centers - points[i]), out=distances, axis=1, dtype=float)
        weights[np.argmin(distances)] += 1

    return weights



# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method SeqWeightedOutliers: sequential k-center with outliers
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def SeqWeightedOutliers(P: List[Tuple], W: List[int], k: int, z: int, alpha: float) -> List[Tuple]:
    return []
#
# ****** ADD THE CODE FOR SeqWeightedOuliers from HW2
#



# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method computeObjective: computes objective function
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def computeObjective(points: RDD, solution: List[Tuple], z: int) -> float :
    return 0.
#
# ****** ADD THE CODE FOR SeqWeightedOuliers from HW2
#


# Just start the main program
if __name__ == "__main__":
    main()


