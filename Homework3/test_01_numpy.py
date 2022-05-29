# Import Packages
from pyspark import SparkConf, SparkContext
import numpy as np
import time
import random
import sys
from typing import List, Tuple, Iterable

from pyspark import RDD # needed fot typing purpuses

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
    #inputPoints = sc.textFile(filename).map(lambda x : strToVector(x)).repartition(L).cache()
    #inputPoints = sc.textFile(filename, L).map(lambda x : strToVector(x), preservesPartitioning=True).cache()  # alternative version which seems to do the same thing
    N = inputPoints.count()
    end = time.time()

    # Pring input parameters
    print("File :" + filename)
    print("Number of points N =", N)
    print("Number of centers k =", k)
    print("Number of outliers z =", z)
    print("Number of partitions L =", L)
    print("Time to read from file:", str((end-start)*1000.), "ms")

    # Solve the problem
    solution = MR_kCenterOutliers(inputPoints, k, z, L)
    
    # Compute the value of the objective function
    start = time.time()
    objective = computeObjective(inputPoints, solution, z)
    end = time.time()
    print("Objective function =", objective)
    print("Time to compute objective function:", str((end-start)*1000.), "ms")
    



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
# Method MR_kCenterOutliers: MR algorithm for k-center with outliers
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def MR_kCenterOutliers(points: RDD, k: int, z: int, L: int) -> List[Tuple[float, ...]]:

    
    #------------- ROUND 1 ---------------------------

    coreset = points.mapPartitions(lambda iterator: extractCoreset(iterator, k+z+1))
    
    # END OF ROUND 1

    
    #------------- ROUND 2 ---------------------------
    start = time.time()
    elems = coreset.collect()
    end = time.time()
    time_round_1 = (end -start) * 1000.

    coresetPoints = []
    coresetWeights = []
    for i in elems:
        coresetPoints.append(i[0])
        coresetWeights.append(i[1])
    
    # ****** ADD YOUR CODE
    # ****** Compute the final solution (run SeqWeightedOutliers with alpha=2)
    # ****** Measure and print times taken by Round 1 and Round 2, separately
    # ****** Return the final solution

    start = time.time()
    S = SeqWeightedOutliers(coresetPoints, coresetWeights, k, z, 2.)
    end = time.time()
    time_round_2 = (end - start) * 1000.

    print("Time Round 1:", time_round_1, "ms")
    print("Time Round 2:", time_round_2, "ms")

    return S
   

# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method extractCoreset: extract a coreset from a given iterator
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def extractCoreset(iter: Iterable[Tuple[float, ...]], points: int) -> List[Tuple]:   # numpy version
    temp = list(iter)   # convert to list before and than to ndarray(needed for some reason(compatibility?))
    partition = np.array(temp, dtype=float)
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
def kCenterFFT(points: np.ndarray, k: int) -> np.ndarray:   # numpy version
    n = points.shape[0] # number of points in dataset
    #random.seed(5000)   # needed for consistency
    idx_rnd = random.randint(0, n-1)
    centers = np.zeros(shape=(k, points.shape[1]), dtype=float)

    centers[0] = points[idx_rnd]

    dist_from_nearest_center = np.sum(a=np.square(points - points[idx_rnd]), axis=1, dtype=float)  #compute square distance between first center and all points (including itself)

    for i in range(k-1):
        centers[i+1] = points[np.argmax(a=dist_from_nearest_center)]    # add to the set of centers the point which is the farthers from the other points in the center

        dist_from_new_center = np.sum(a=np.square(points - centers[i+1]), axis=1, dtype=float)
        np.minimum(dist_from_nearest_center, dist_from_new_center, out=dist_from_nearest_center)

    return centers

# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method computeWeights: compute weights of coreset points
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def computeWeights(points: np.ndarray, centers: np.ndarray) -> np.ndarray:
    weights = np.zeros(shape=centers.shape[0], dtype=int)   # init wheights array

    distances = np.zeros(shape=centers.shape[0], dtype=float)
    for i in range(points.shape[0]):
        np.sum(a=np.square(centers - points[i]), out=distances, axis=1, dtype=float)    # store in "distances" the distance between a point and each fft center
        weights[np.argmin(distances)] += 1      # add one unit of weight to

    return weights



# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method SeqWeightedOutliers: sequential k-center with outliers
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def SeqWeightedOutliers(P: List[Tuple], W: List[int], k: int, z: int, alpha: float) -> List[Tuple]:

    # convert list of tuples into ndarrays
    P_np = np.array(P, dtype=float)
    W = np.array(W, dtype=int)

    n, dims = P_np.shape[0], P_np.shape[1]  # used to make the core more readable
    attempts = 0

    # calculate array containing the distance between all points squared
    # (because when we need to compare the data it's possible to save little time by not doing the square root)
    all_dist_squared = np.zeros(shape=(n,n), dtype=P_np.dtype)
    for i in range(n):
        np.sum(a=np.square(P_np - P_np[i]), out=all_dist_squared[i], axis=1, dtype=P_np.dtype)

    # compute and print first guess
    guess_samples = k + z + 1
    r_map_matr = np.zeros(shape=(n,n), dtype=np.bool8)  # needed to avoid considering the diagonal(which is all zeros) in the computation of the minimum
    r_map_matr[:guess_samples, :guess_samples] = all_dist_squared[:guess_samples, :guess_samples]

    #r_squared = all_dist_squared[r_map_matr].min() / 4.  # because it is squared so r^2 / 4 = (r/2)^2
    r_squared = np.square(0.008139410298050962)

    print("Initial guess =", np.sqrt(r_squared))

    while True:
        S = np.zeros(shape=(k, dims), dtype=P_np.dtype)
        iter_weights = np.copy(a=W) # every time it covers new points the weight of such points get set to 0 so that they will be ignored in the next iteration

        for i in range(k):
            best_weight = 0.
            best_pt_id = -1
            ball_radius_squared = np.square(1.+2.*alpha)*r_squared # used when selecting the center

            for current_pt in range(n):
                if iter_weights[current_pt] > 0:    # since every covered point gets its weight value inside of iter_weight changed to 0 we can assume a new center won't be one of such values(better performance)
                    current_weight = np.sum(a=iter_weights[all_dist_squared[current_pt] < ball_radius_squared], dtype=float)
                    if current_weight > best_weight:
                        best_weight = current_weight
                        best_pt_id = current_pt

            if best_pt_id == -1:    # this case is the one where the best_pt_id did not change at all during the run of the algorithm, which means that all the points have already been covered
                S = S[:i]   # generate new ndarray which size matches the number of centers found
                break
            S[i] = P[best_pt_id]    # add new center
            ball_radius_squared = np.square(3.+4.*alpha)*r_squared # used when removing new covered points
            iter_weights[all_dist_squared[best_pt_id] < ball_radius_squared] = 0.

        outliers_w = np.sum(a=iter_weights) # sum all weights of the points that are still uncovered
        attempts += 1

        if outliers_w <= z:
            print("Final guess =", np.sqrt(r_squared))
            print("Number of guesses =", attempts)
            # convert S to list of tuples before returning it
            S = list(map(tuple, S))
            return S
        else:
            r_squared *= 4. # because it is squared so r^2 * 4 = (r*2)^2



# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method computeObjective: computes objective function
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def computeObjective(points: RDD, solution: List[Tuple], z: int) -> float :
    
    intermediateRDD = points.mapPartitions(lambda iterator: find_max_Z_plus_one_points(iter=iterator, toKeep=(z+1), solution=solution))

    elem = intermediateRDD.collect()

    farthest_points = np.array(elem)

    # removing outliers
    for i in range(z):
        # setting to zero is faster than remoiving an element from ndarray(because removing and element involves reshaping the whole ndarray)
        farthest_points[np.argmax(a=farthest_points)] = 0.
    
    return np.sqrt(np.max(farthest_points))   # sqrt needed because, like in the rest of the algorithm, all the distances are squared


def find_max_Z_plus_one_points(iter: Iterable, toKeep: int, solution: List[Tuple[float, ...]]) -> List[float]:

    temp = list(iter)
    part_pts = np.array(temp)
    sol = np.array(solution)
    n, k = part_pts.shape[0], sol.shape[0]

    dist_from_centers = np.zeros(shape=(n,k), dtype= float)
    for i in range(n):
        np.sum(np.square(np.subtract(part_pts[i], sol)), out=dist_from_centers[i], axis=1, dtype=float)

    min_dist_from_centers = dist_from_centers.min(axis=1)   # stores the distance between each point and the closest solution to it

    part_max_dists = min_dist_from_centers[min_dist_from_centers.argsort()[n-toKeep:]]  # stores the biggest toKeep's values from min_dist_from_centers

    return list(part_max_dists)


# Just start the main program
if __name__ == "__main__":
    main()


