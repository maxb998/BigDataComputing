# Import Packages
from pyspark import SparkConf, SparkContext
import numpy as np
import time
import random
import sys
import math
from typing import List, Tuple

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
    start = 0
    end = 0

    # Set Spark Configuration
    conf = SparkConf().setAppName('MR k-center with outliers')
    sc = SparkContext(conf=conf)
    sc.setLogLevel("WARN")

    # Read points from file
    start = time.time()
    inputPoints = sc.textFile(filename, L).map(lambda x : strToVector(x)).repartition(L).cache()
    N = inputPoints.count()
    end = time.time()

    # Pring input parameters
    print("File : " + filename)
    print("Number of points N = ", N)
    print("Number of centers k = ", k)
    print("Number of outliers z = ", z)
    print("Number of partitions L = ", L)
    print("Time to read from file: ", str((end-start)*1000), " ms")

    # Solve the problem
    solution = MR_kCenterOutliers(inputPoints, k, z, L)
    print(solution)

    # Compute the value of the objective function
    #start = time.time()
    #objective = computeObjective(inputPoints, solution, z)
    #end = time.time()
    #print("Objective function = ", objective)
    #print("Time to compute objective function: ", str((end-start)*1000), " ms")
     



# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# AUXILIARY METHODS
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method strToVector: input reading
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def strToVector(str):
    out = tuple(map(float, str.split(',')))
    return out



# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method squaredEuclidean: squared euclidean distance
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def squaredEuclidean(point1,point2):
    res = 0
    for i in range(len(point1)):
        diff = (point1[i]-point2[i])
        res +=  diff*diff
    return res



# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method euclidean:  euclidean distance
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def euclidean(point1,point2):
    res = 0
    for i in range(len(point1)):
        diff = (point1[i]-point2[i])
        res +=  diff*diff
    return math.sqrt(res)



# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method MR_kCenterOutliers: MR algorithm for k-center with outliers
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def MR_kCenterOutliers(points, k, z, L):

    
    #------------- ROUND 1 ---------------------------

    coreset = points.mapPartitions(lambda iterator: extractCoreset(iterator, k+z+1))
    
    # END OF ROUND 1

    
    #------------- ROUND 2 ---------------------------
    
    start = time.time()
    elems = coreset.collect()
    end = time.time()
    print("Time to extract coreset: ", str((end-start)*1000.), " ms")

    coresetPoints = list()
    coresetWeights = list()
    for i in elems:
        coresetPoints.append(i[0])
        coresetWeights.append(i[1])
    
    # ****** ADD YOUR CODE
    # ****** Compute the final solution (run SeqWeightedOutliers with alpha=2)
    # ****** Measure and print times taken by Round 1 and Round 2, separately
    # ****** Return the final solution
    S = SeqWeightedOutliers(coresetPoints, coresetWeights, k, z, 2)     
    return S     
   

# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method extractCoreset: extract a coreset from a given iterator
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def extractCoreset(iter, points):
    partition = list(iter)
    centers = kCenterFFT(partition, points)
    weights = computeWeights(partition, centers)
    c_w = list()
    for i in range(0, len(centers)):
        entry = (centers[i], weights[i])
        c_w.append(entry)
    # return weighted coreset
    return c_w
    
    
    
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method kCenterFFT: Farthest-First Traversal
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def kCenterFFT(points, k):
    random.seed(5000)
    idx_rnd = random.randint(0, len(points)-1)
    centers = [points[idx_rnd]]
    related_center_idx = [idx_rnd for i in range(len(points))]
    dist_near_center = [squaredEuclidean(points[i], centers[0]) for i in range(len(points))]

    for i in range(k-1):
        new_center_idx = max(enumerate(dist_near_center), key=lambda x: x[1])[0] # argmax operation
        centers.append(points[new_center_idx])
        for j in range(len(points)):
            if j != new_center_idx:
                dist = squaredEuclidean(points[j], centers[-1])
                if dist < dist_near_center[j]:
                    dist_near_center[j] = dist
                    related_center_idx[j] = new_center_idx
            else:
                dist_near_center[j] = 0
                related_center_idx[j] = new_center_idx
    return centers



# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method computeWeights: compute weights of coreset points
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def computeWeights(points, centers):
    weights = np.zeros(len(centers))
    for point in points:
        mycenter = 0
        mindist = squaredEuclidean(point,centers[0])
        for i in range(1, len(centers)):
            dist = squaredEuclidean(point,centers[i])
            if dist < mindist:
                mindist = dist
                mycenter = i
        weights[mycenter] = weights[mycenter] + 1
    return weights



# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method SeqWeightedOutliers: sequential k-center with outliers
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def SeqWeightedOutliers(P: List[Tuple], W: List[int], k: int, z: int, alpha: float) -> List[Tuple]:

    # convert list of tuples into ndarrays
    P_np: np.ndarray = np.array(P, dtype=float)
    W: np.ndarray = np.array(W, dtype=int)

    n, dims = P_np.shape[0], P_np.shape[1]  # used to make the core more readable
    attempts: int = 0

    # calculate array containing the distance between all points squared
    # (because when we need to compare the data it's possible to save little time by not doing the square root)
    all_dist_squared: np.ndarray = np.zeros(shape=(n,n), dtype=P_np.dtype)
    for i in range(n):
        np.sum(a=np.square(P_np - P_np[i]), out=all_dist_squared[i], axis=1, dtype=P_np.dtype)

    # compute and print first guess
    guess_samples: int = k + z + 1
    r_map_matr: np.ndarray = np.zeros(shape=(n,n), dtype=np.bool8)  # needed to avoid considering the diagonal(which is all zeros) in the computation of the minimum
    r_map_matr[:guess_samples, :guess_samples] = all_dist_squared[:guess_samples, :guess_samples]
    r_squared: float = all_dist_squared.min(initial=all_dist_squared[0,1], where=r_map_matr) / 4.  # because it is squared so r^2 / 4 = (r/2)^2
    print("Initial guess = ", np.sqrt(r_squared))

    while True:
        S: np.ndarray = np.zeros(shape=(k, dims), dtype=P_np.dtype)
        iter_weights: np.ndarray = np.copy(a=W) # every time it covers new points the weight of such points get set to 0 so that they will be ignored in the next iteration

        for i in range(k):
            best_weight: float = 0.
            best_pt_id: int = -1
            ball_radius_squared: float = np.square(1.+2.*alpha)*r_squared # used when selecting the center

            for current_pt in range(n):
                if iter_weights[current_pt] > 0:    # since every covered point gets its weight value inside of iter_weight changed to 0 we can assume a new center won't be one of such values(better performance)
                    current_weight = np.sum(a=iter_weights, dtype=float, where=all_dist_squared[current_pt] < ball_radius_squared)
                    if current_weight > best_weight:
                        best_weight = current_weight
                        best_pt_id = current_pt

            if best_pt_id == -1:    # this case is the one where the best_pt_id did not change at all during the run of the algorithm, which means that all the points have already been covered
                S: np.ndarray = S[:i]   # generate new ndarray which size matches the number of centers found
                break
            S[i] = P[best_pt_id]    # add new center
            ball_radius_squared: float = np.square(3.+4.*alpha)*r_squared # used when removing new covered points
            iter_weights[all_dist_squared[best_pt_id] < ball_radius_squared] = 0.

        outliers_w = np.sum(a=iter_weights) # sum all weights of the points that are still uncovered
        attempts += 1

        if outliers_w <= z:
            print("Final guess = ", np.sqrt(r_squared))
            print("Number of guesses = ", attempts)
            # convert S to list of tuples before returning it
            S: list[tuple] = list(map(tuple, S))
            return S
        else:
            r_squared *= 4. # because it is squared so r^2 * 4 = (r*2)^2



# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method computeObjective: computes objective function
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def computeObjective(inputPoints: List[Tuple], solution: List[Tuple], z: int) -> float :

    # convert list of tuples into ndarrays
    sol: np.ndarray = np.array(object=solution, dtype=float)
    inputPoints_np: np.ndarray = np.array(object=inputPoints, dtype=float)

    n, k = len(inputPoints), len(solution)

    # compute distances for each point, between the point itself and all the centers, result is a matrix n*k
    dist_from_centers: np.ndarray = np.zeros(shape=(n,k), dtype=float)
    for i in range(n):
        dist_from_centers[i] = np.sum(a=np.square(np.subtract(sol, inputPoints_np[i])), axis=1, dtype=float)
    
    min_dist_from_centers: np.ndarray = dist_from_centers.min(axis=1)   # stores the distance between each point and the closest solution to it

    # removing outliers
    for i in range(z):
        # setting to zero is faster than remoiving an element from ndarray(because removing and element involves reshaping the whole ndarray)
        min_dist_from_centers[np.argmax(a=min_dist_from_centers)] = 0.
    
    return np.sqrt(np.max(min_dist_from_centers))   # sqrt needed because, like in the rest of the algorithm, all the distances are squared




# Just start the main program
if __name__ == "__main__":
    main()


