import sys
import os
import time
import numpy as np


def SeqWeightedOutliers(P: np.ndarray, W: np.ndarray, k: int, z: int, alpha: np.float64) -> np.ndarray:

    n, dims = P.shape[0], P.shape[1]
    attempts: int = 0

    # calculate array containing the distance between all points squared(because when we need to compare the data it's possible to save little time by not doing the square root)
    all_dist_squared: np.ndarray = np.zeros(shape=(n,n), dtype=P.dtype)
    for i in range(n):
        coord_diff_squared: np.ndarray = np.square(P - P[i])
        all_dist_squared[i] = np.sum(a=coord_diff_squared, axis=1, dtype=P.dtype)

    # calculate first guess
    guess_samples: int = k + z + 1
    r_map_matr: np.ndarray = np.zeros(shape=(n,n), dtype=np.bool8)
    r_map_matr[:guess_samples, :guess_samples] = all_dist_squared[:guess_samples, :guess_samples]
    r_squared: np.float64 = all_dist_squared.min(initial=all_dist_squared[0,1], where=r_map_matr) / 4.  # because it is squared so r^2 / 4 = (r/2)^2
    
    print("Initial guess = ", np.sqrt(r_squared))

    while True:
        S: np.ndarray = np.zeros(shape=(k, dims), dtype=P.dtype)
        is_uncovered: np.ndarray = np.ones(shape=n, dtype=np.bool8)
        ball_radius_squared: np.float64 = (1.+2.*alpha)*(1.+2.*alpha)*r_squared # used when selecting the center

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
            ball_radius_squared: np.float64 = (3.+4.*alpha)*(3.+4.*alpha)*r_squared #used when removing new covered points
            is_uncovered[all_dist_squared[best_pt_id] < ball_radius_squared] = False

        outliers_w = np.sum(a=W, where=is_uncovered)
        attempts += 1
        if outliers_w <= z:
            print("Final guess = ", np.sqrt(r_squared))
            print("Number of guesses = ", attempts)
            return S
        else:
            r_squared *= 4. # because it is squared so r^2 * 4 = (r*2)^2


def ComputeObjective(inputPoints: np.ndarray, solution: np.ndarray, z: int) -> np.float64 :
    n, k = inputPoints.shape[0], solution.shape[0]
    dist_from_centers: np.ndarray = np.zeros(shape=(n,k), dtype=inputPoints.dtype)

    for i in range(n):
        coord_diff_squared: np.ndarray = np.square(np.subtract(solution, inputPoints[i]))   # shape=(k,dims)
        dist_from_centers[i] = np.sum(a=coord_diff_squared, axis=1, dtype=inputPoints.dtype)
        #dist_from_centers[i] = np.sum(a=np.square(np.subtract(solution, inputPoints[i])), axis=1, dtype=inputPoints.dtype)
    
    min_dist_from_centers: np.ndarray = dist_from_centers.min(axis=1)

    for i in range(z):
        min_dist_from_centers[np.argmax(a=min_dist_from_centers)] = 0.
            
    return np.sqrt(min_dist_from_centers[np.argmax(a=min_dist_from_centers)])


def main():
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

    # print input data informations
    print("Input size n = ", data.shape[0])
    print("Number of centers k = ", k)
    print("Number of outliers z = ", z)

    # do k-center with weight and outliers
    millis_start: float = time.time() * 1000.
    centers = SeqWeightedOutliers(P=data, W=weights, k=k, z=z, alpha=alpha)
    millis_end: float = time.time() * 1000.

    # calculate time needed in milliseconds
    millis_duration: float = millis_end - millis_start

    # Compute objective
    obj: np.float64 = ComputeObjective(inputPoints=data, solution=centers, z=z)

    # print output
    print("Objective function = ", obj)
    print("Time of SeqWeightedOutliers = ", millis_duration)


if __name__ == "__main__":
    main()