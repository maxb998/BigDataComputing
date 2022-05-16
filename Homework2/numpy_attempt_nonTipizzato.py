import sys
import os
import time
import numpy as np


def SeqWeightedOutliers(P, W, k, z, alpha):

    n, dims = P.shape[0], P.shape[1]
    attempts = 0

    # calculate array containing the distance between all points squared
    # (because when we need to compare the data it's possible to save little time by not doing the square root)
    all_dist_squared = np.zeros(shape=(n,n), dtype=P.dtype)
    for i in range(n):
        all_dist_squared[i] = np.sum(a=np.square(P - P[i]), axis=1, dtype=P.dtype)

    # compute and print first guess
    guess_samples = k + z + 1
    r_map_matr = np.zeros(shape=(n,n), dtype=np.bool8)  # needed to avoid considering the diagonal(which is all zeros) in the computation of the minimum
    r_map_matr[:guess_samples, :guess_samples] = all_dist_squared[:guess_samples, :guess_samples]
    r_squared = all_dist_squared.min(initial=all_dist_squared[0,1], where=r_map_matr) / 4.  # because it is squared so r^2 / 4 = (r/2)^2
    print("Initial guess = ", np.sqrt(r_squared))

    while True:
        S = np.zeros(shape=(k, dims), dtype=P.dtype)
        iter_weights = np.copy(a=W) # every time it covers new points the weight of such points get set to 0 so that they will be ignored in the next iteration

        for i in range(k):
            best_weight = 0.
            best_pt_id = -1
            ball_radius_squared = np.square(1.+2.*alpha)*r_squared # used when selecting the center

            for current_pt in range(n):
                if iter_weights[current_pt] > 0:    # since every covered point gets its weight value inside of iter_weight changed to 0 we can assume a new center won't be one of such values(better performance)
                    current_weight = np.sum(a=iter_weights, dtype=float, where=all_dist_squared[current_pt] < ball_radius_squared)
                    if current_weight > best_weight:
                        best_weight = current_weight
                        best_pt_id = current_pt

            # *** SPECIAL CASE -> must ask the professor: if the ball radius increase "too much" between and iteration and the other
            # it might happen that all points get inside a number of clusters which is less than k, therefore there will be some cluster
            # left with no points, completely empty, non-existent. This if condition and its content just give a warning about the fact
            # that the special case occurred and prevents the algorithm from giving "fake" clusters.
            # A proper solution to this would probably be to enter an algorithm to adjust the radius of the ball in order to get a better one
            # something like a binary search between the previous radius and the current one.
            # It must also be noted that this type of solution may potentially increase the number of attempts of the algorithm to get the correct solution
            # It also must be noted that probably this is the way the pseudocode works
            if best_pt_id == -1:    
                #print("Could not find ", k, "centers. Only ", i, "found")
                S = S[:i]   # generate new ndarray which size matches the number of centers found
                break
            S[i] = P[best_pt_id]    # add new center
            ball_radius_squared = np.square(3.+4.*alpha)*r_squared # used when removing new covered points
            iter_weights[all_dist_squared[best_pt_id] < ball_radius_squared] = 0.

        outliers_w = np.sum(a=iter_weights) # sum all weights of the points that are still uncovered
        attempts += 1

        if outliers_w <= z:
            print("Final guess = ", np.sqrt(r_squared))
            print("Number of guesses = ", attempts)
            return S
        else:
            r_squared *= 4. # because it is squared so r^2 * 4 = (r*2)^2


def ComputeObjective(inputPoints, solution, z) -> float :
    n, k = inputPoints.shape[0], solution.shape[0]

    # compute distances for each point, between the point itself and all the centers, result is a matrix n*k
    dist_from_centers = np.zeros(shape=(n,k), dtype=inputPoints.dtype)
    for i in range(n):
        dist_from_centers[i] = np.sum(a=np.square(np.subtract(solution, inputPoints[i])), axis=1, dtype=inputPoints.dtype)
    
    min_dist_from_centers = dist_from_centers.min(axis=1)   # stores the distance between each point and the closest solution to it

    # removing outliers
    for i in range(z):
        # setting to zero is faster than remoiving an element from ndarray(because removing and element involves reshaping the whole ndarray)
        min_dist_from_centers[np.argmax(a=min_dist_from_centers)] = 0.
    
    return np.sqrt(np.max(min_dist_from_centers))   # sqrt needed because, like in the rest of the algorithm, all the distances are squared


def main():
    # np.set_printoptions(precision=2, linewidth=300)   # useful when printing ndarrays during debugging
    # Check argv lenght and content
    assert len(sys.argv) == 4, "Usage: <Filename> <K> <Z>"

    filename = sys.argv[1]
    assert os.path.isfile(filename), "File or folder not found"
    try:
        f = open(filename)
        inputPoints = np.loadtxt(fname=filename, dtype=float, delimiter=',')
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
    weights = np.ones(shape=inputPoints.shape[0], dtype=int)

    # alpha
    alpha = 0.

    # print input data informations
    print("Input size n = ", inputPoints.shape[0])
    print("Number of centers k = ", k)
    print("Number of outliers z = ", z)

    # do k-center with weight and outliers
    millis_start = time.time() * 1000.
    solution = SeqWeightedOutliers(P=inputPoints, W=weights, k=k, z=z, alpha=alpha)
    millis_end = time.time() * 1000.

    # calculate time needed in milliseconds
    millis_duration = millis_end - millis_start

    # Compute objective
    objective = ComputeObjective(inputPoints=inputPoints, solution=solution, z=z)

    # print output
    print("Objective function = ", objective)
    print("Time of SeqWeightedOutliers = ", millis_duration)

if __name__ == "__main__":
    main()