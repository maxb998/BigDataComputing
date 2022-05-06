import sys
import os
import time
import numpy as np


def calculate_r(points: np.ndarray, n: int) -> np.float64:
    n = n-1
    reduced_points: np.ndarray = points[:n]
    distances: np.ndarray = np.zeros(shape=(n, n), dtype=points.dtype)

    for i in range(n):
        distances[i] = np.linalg.norm(x=(reduced_points-reduced_points[i]), axis=-1)
    distances = distances[~np.eye(n,dtype=bool)].reshape(n,-1)  # remove diagonal which is all zeros

    return (distances.min() / 2.)


def SeqWeightedOutliers(P: np.ndarray, W: np.ndarray, k: int, z: int, alpha: np.float64) -> tuple[np.ndarray, int, np.float64, np.float64]:

    n, dims = P.shape[0], P.shape[1]
    range_n: np.ndarray = np.arange(n)
    attempts: int = 0

    # calculate array containing the distance between all points squared(because when we need to compare the data it's possible to save little time by not doing the square root)
    all_dist_squared: np.ndarray = np.zeros(shape=(n,n), dtype=P.dtype)
    for i in range(n):
        coord_diff_squared: np.ndarray = np.square(P - P[i])
        all_dist_squared[i] = np.sum(a=coord_diff_squared, axis=1, dtype=P.dtype)
    #all_distances_squared = all_distances_squared[~np.eye(n,dtype=bool)].reshape(n,-1)  # remove diagonal which is all zeros

    r: np.float64 = calculate_r(P, k+z+1)
    first_r: np.float64 = r

    while True:
        S: np.ndarray = np.zeros(shape=(k, dims), dtype=P.dtype)
        is_uncovered: np.ndarray = np.ones(shape=n, dtype=np.bool8)
        ball_radius_squared: np.float64 = ((1.+2.*alpha)*r)**2 # used when selecting the center

        for i in range(k):
            best_weight: np.float64 = 0.
            best_pt_id: int = 0

            for current_pt in range(n):
                if is_uncovered[current_pt]:
                    current_weight = np.sum(a=W, dtype=np.float64, where=all_dist_squared[current_pt] < ball_radius_squared)
                    if current_weight > best_weight:
                        best_weight = current_weight
                        best_pt_id = current_pt

            S[i] = P[best_pt_id]    # add new center
            ball_radius_squared: np.float64 = ((3.+4.*alpha)*r)**2 #used when removing new covered points
            is_uncovered[all_dist_squared[best_pt_id] < ball_radius_squared] = False

        outliers_w = np.sum(a=W, where=is_uncovered)
        attempts += 1
        if outliers_w <= z:
            return S, attempts, first_r, r
        else:
            r = r * 2.

def ComputeObjective(inputPoints: np.ndarray, solution: np.ndarray, weights: np.ndarray, z: int) -> np.float64 :
    n, dims, k = inputPoints.shape[0], inputPoints.shape[1], solution.shape[0]
    dist_from_centers: np.ndarray = np.zeros(shape=(n,k), dtype=inputPoints.dtype)

    for i in range(n):
        coord_diff_squared: np.ndarray = np.square(inputPoints[i] - solution)
        dist_from_centers[i] = np.sum(a=coord_diff_squared, axis=1, dtype=inputPoints.dtype)
    
    min_dist_from_centers: np.ndarray = dist_from_centers.min(axis=1)
    sorted_indexs: np.ndarray = np.argsort(a=min_dist_from_centers)

    pos: int = -1
    weight_sum: int = 0
    for i in range(n-1, 0, -1):
        weight_sum += weights[sorted_indexs[i]]
        if weight_sum > z:
            pos: int = sorted_indexs[i]
            break
            

    return np.sqrt(min_dist_from_centers[pos])

def main():
    np.set_printoptions(precision=2, linewidth=200)

    # Check argv lenght and content
    assert len(sys.argv) == 4, "Usage: <Filename> <K> <Z>"

    filename = sys.argv[1]
    assert os.path.isfile(filename), "File or folder not found"
    try:
        f = open(filename)
        data: np.ndarray = np.loadtxt(fname=filename, dtype=np.float64, delimiter=',')
    finally:
        f.close()
    
    k = sys.argv[2]
    assert k.isdigit(), "K must be an integer value"
    k = int(k)
    assert k >= 2, "K must be at least 2"

    z = sys.argv[3]
    assert z.isdigit(), "Z must be an integer value"
    z = int(z)
    assert z >= 0, "K must be positive"

    # init unit weights
    weights: np.ndarray = np.ones(shape=data.shape[0], dtype=np.int64)

    # alpha
    alpha: np.float64 = 0.

    # do k-center with weight and outliers
    millis_start: float = time.time() * 1000.
    centers, attemps, first_guess, final_guess = SeqWeightedOutliers(P=data, W=weights, k=k, z=z, alpha=alpha)
    millis_end: float = time.time() * 1000.

    # calculate time needed in milliseconds
    millis_duration: float = millis_end - millis_start

    # Compute objective
    obj: np.float64 = ComputeObjective(inputPoints=data, solution=centers, weights=weights, z=z)

    # print output
    print("Input size n = ", data.shape[0])
    print("Number of centers k = ", k)
    print("Number of outliers z = ", z)
    print("Initial guess = ", first_guess)
    print("Final guess = ", final_guess)
    print("Number of guesses = ", attemps)
    print("Objective function = ", obj)
    print("Time of SeqWeightedOutliers = ", millis_duration)


if __name__ == "__main__":
    main()