# Import Packages
from pyspark import SparkConf, SparkContext
import numpy as np
import time
import random
import sys
from typing import List, Tuple, Iterable

from pyspark import RDD # needed fot typing purpuses(can be removed if the all the typos 'RDD' are removed from parameters in function definitions)

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

    # Print input parameters
    print("File :" + filename)
    print("Number of points N =", N)
    print("Number of centers k =", k)
    print("Number of outliers z =", z)
    print("Number of partitions L =", L)
    print("Time to read from file:", str((end-start)*1000.), "ms")

    # Solve the problem
    solution = MR_kCenterOutliers(inputPoints, k, z, L, N)
    
    # Compute the value of the objective function
    start = time.time()
    objective = computeObjective(inputPoints, solution, z, N, L)
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
# Method 
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def loadNdarrayFromIterable(iter: Iterable, start_arr_size: int) -> np.ndarray:
    
    # read and save the first element to get the number of dimension of the points in the dataset
    for elem in iter:
        first = tuple(elem)
        break
    dims = len(first)
    
    arr = np.zeros(shape=(int(start_arr_size*1.1),dims), dtype=np.float64)  # init container of all points
    arr[0] = first  # write in the proper array the first element
    del(first)  # release ram
    n = 1   # init counter for the number of elements inside the partition

    # now to fill the array with the rdd partition data
    for elem in iter:
        arr[n] = tuple(elem)
        n += 1
        if n >= arr.shape[0]:   # check if the array is full
            arr.resize((int(arr.shape[0]*1.1),dims))    # if array is full: increase its size
    
    arr.resize(n,dims)  # resize array to fit the data collected
    return arr
    

# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method MR_kCenterOutliers: MR algorithm for k-center with outliers
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def MR_kCenterOutliers(points: RDD, k: int, z: int, L: int, n: int) -> List[Tuple]:

    
    #------------- ROUND 1 ---------------------------

    coreset = points.mapPartitions(lambda iterator: extractCoreset(iter=iterator, points=(k+z+1), expected_num_of_pts=(n/L)))
    
    # END OF ROUND 1

    
    #------------- ROUND 2 ---------------------------
    start = time.time()
    elems = coreset.collect()
    end = time.time()
    time_round_1 = (end -start) * 1000.

    start = time.time()     # round 2 start time measure
    coresetPoints = []
    coresetWeights = []
    for i in elems:
        coresetPoints.append(i[0])
        coresetWeights.append(i[1])
    
    # ****** ADD YOUR CODE
    # ****** Compute the final solution (run SeqWeightedOutliers with alpha=2)
    # ****** Measure and print times taken by Round 1 and Round 2, separately
    # ****** Return the final solution

    S = SeqWeightedOutliers(P=coresetPoints, W=coresetWeights, k=k, z=z, alpha=2.)

    end = time.time()       # round 2 end time measure
    time_round_2 = (end - start) * 1000.

    print("Time Round 1:", time_round_1, "ms")
    print("Time Round 2:", time_round_2, "ms")

    return S
   

# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method extractCoreset: extract a coreset from a given iterator
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def extractCoreset(iter: Iterable, points: int, expected_num_of_pts: int) -> List[Tuple]:   # numpy version

    partitionPts = loadNdarrayFromIterable(iter=iter, start_arr_size=expected_num_of_pts)
    #partitionPts = np.array(list(iter), dtype=np.float64)  # this one is faster but requires double ram space which is an issue with the largest dataset(HIGGS10M7D)
    centers = kCenterFFT(partitionPts, points)
    weights = computeWeights(partitionPts, centers)
    
    
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

    dist_from_nearest_center = np.sum(a=np.square(points - points[idx_rnd]), axis=1, dtype=np.float64)  #compute square distance between first center and all points (including itself)

    for i in range(k-1):
        centers[i+1] = points[np.argmax(a=dist_from_nearest_center)]    # add to the set of centers the point which is the farthers from the other points in the center

        dist_from_new_center = np.sum(a=np.square(points - centers[i+1]), axis=1, dtype=np.float64)
        np.minimum(dist_from_nearest_center, dist_from_new_center, out=dist_from_nearest_center)

    return centers

# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method computeWeights: compute weights of coreset points
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def computeWeights(points: np.ndarray, centers: np.ndarray) -> np.ndarray:
    weights = np.zeros(shape=centers.shape[0], dtype=int)   # init wheights array(size equal to the number of centers)

    distances = np.zeros(shape=centers.shape[0], dtype=np.float64)
    for i in range(points.shape[0]):
        np.sum(a=np.square(centers - points[i]), out=distances, axis=1, dtype=np.float64)    # store in "distances" the distance between a point and each fft center
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

    r_squared = all_dist_squared[r_map_matr].min() / 4.  # because it is squared so r^2 / 4 = (r/2)^2

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
                    current_weight = np.sum(a=iter_weights[all_dist_squared[current_pt] < ball_radius_squared], dtype=int)
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
def computeObjective(points: RDD, solution: List[Tuple], z: int, n: int, L: int) -> float :
    
    intermediateRDD = points.mapPartitions(lambda iterator: find_max_Z_plus_one_points(iter=iterator, distsToKeep=(z+1), solution=solution, expected_num_of_pts=(n/L)))

    farthest_points = np.array(intermediateRDD.collect())

    # removing outliers
    for i in range(z):
        # setting to zero is faster than remoiving an element from ndarray(because removing and element involves reshaping the whole ndarray)
        farthest_points[np.argmax(a=farthest_points)] = 0.
    
    return np.sqrt(np.max(farthest_points))   # sqrt needed because, like in the rest of the algorithm, all the distances are squared


def find_max_Z_plus_one_points(iter: Iterable, distsToKeep: int, solution: List[Tuple[float, ...]], expected_num_of_pts: int) -> List[float]:

    partitionPts = loadNdarrayFromIterable(iter=iter, start_arr_size=expected_num_of_pts)
    #partitionPts = np.array(list(iter))    # this one is faster but requires double ram space which is an issue with the largest dataset(HIGGS10M7D)

    sol = np.array(solution)
    n, k = partitionPts.shape[0], sol.shape[0]

    dist_from_centers = np.zeros(shape=(n,k), dtype=np.float64)
    for i in range(n):
        np.sum(np.square(np.subtract(partitionPts[i], sol)), out=dist_from_centers[i], axis=1, dtype=dist_from_centers.dtype)

    min_dist_from_centers = dist_from_centers.min(axis=1)   # stores the distance between each point and the closest solution to it

    #part_max_dists = min_dist_from_centers[min_dist_from_centers.argsort()[n-distsToKeep:]]  # stores the biggest distsToKeep's values from min_dist_from_centers ***POSSIBLY SORTS AN UNECESSARY NUMBER OF ITEMS 
    #return list(part_max_dists)

    maxDists = np.zeros(shape=distsToKeep, dtype=np.float64)
    for i in range(distsToKeep):
        max_id = min_dist_from_centers.argmax()
        maxDists[i] = min_dist_from_centers[max_id]
        min_dist_from_centers[max_id] = 0.
    return list(maxDists)


# Just start the main program
if __name__ == "__main__":
    main()


