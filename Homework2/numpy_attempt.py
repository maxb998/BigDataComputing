import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt


def calculate_r(points: np.ndarray, n: int) -> np.float32:
    n = n-1
    reduced_points: np.ndarray = points[:n]
    distances: np.ndarray = np.zeros(shape=(n, n), dtype=points.dtype)

    for i in range(n):
        distances[i] = np.linalg.norm(x=(reduced_points-reduced_points[i]), axis=-1)
    distances = distances[~np.eye(n,dtype=bool)].reshape(n,-1)  # remove diagonal which is all zeros

    #print(distances)
    #print(distances.min())

    return (distances.min() / 2.)


def SeqWeightedOutliers(P: np.ndarray, W: np.ndarray, k: int, z: int, alpha: np.float32) -> tuple[np.ndarray, int, np.float32, np.float32]:

    n, dims = P.shape[0], P.shape[1]
    range_n: np.ndarray = np.arange(n)
    attempts: int = 0

    # calculate array containing the distance between all points squared(because when we need to compare the data it's possible to save little time by not doing the square root)
    all_dist_squared: np.ndarray = np.zeros(shape=(n,n), dtype=P.dtype)
    for i in range(n):
        coord_diff_squared: np.ndarray = np.square(P - P[i])
        all_dist_squared[i] = np.sum(a=coord_diff_squared, axis=1, dtype=P.dtype)
    #all_distances_squared = all_distances_squared[~np.eye(n,dtype=bool)].reshape(n,-1)  # remove diagonal which is all zeros

    r: np.float32 = calculate_r(P, k+z+1)
    first_r: np.float32 = np.sqrt(r)

    while True:
        S: np.ndarray = np.zeros(shape=(k, dims), dtype=P.dtype)
        is_uncovered: np.ndarray = np.ones(shape=n, dtype=np.bool8)
        ball_radius_squared: np.float32 = ((1.+2.*alpha)*r)**2 # used when selecting the center
        '''
        print(all_dist_squared[0])
        print(all_dist_squared[:,0])
        print()
        '''
        for i in range(k):
            best_weight: np.float32 = 0.
            best_pt_id: int = 0

            for current_pt in range(n):
                if is_uncovered[current_pt]:
                    current_weight = np.sum(a=W, dtype=np.float32, where=all_dist_squared[current_pt] < ball_radius_squared)
                    if current_weight > best_weight:
                        best_weight = current_weight
                        best_pt_id = current_pt
            '''
            print(best_pt_id)
            print(best_weight)
            print()
            '''
            S[i] = P[best_pt_id]    # add new center
            ball_radius_squared: np.float32 = ((3.+4.*alpha)*r)**2 #used when removing new covered points
            is_uncovered[all_dist_squared[best_pt_id] < ball_radius_squared] = False
            #print(is_uncovered.shape)
            #print((all_dist_squared[best_pt_id] < ball_radius_squared))

        outliers_w = np.sum(a=W, where=is_uncovered)
        #print(outliers_w)
        #print(is_uncovered == 1)
        attempts += 1
        if outliers_w <= z:
            return S, attempts, first_r, np.sqrt(r)
        else:
            r = r * 2.



def main():
    np.set_printoptions(precision=2, linewidth=200)

    # Check argv lenght and content
    assert len(sys.argv) == 4, "Usage: <Filename> <K> <Z>"

    filename = sys.argv[1]
    assert os.path.isfile(filename), "File or folder not found"
    try:
        f = open(filename)
        data: np.ndarray = np.loadtxt(fname=filename, dtype=np.float32, delimiter=',')
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
    alpha: np.float32 = 0.

    # do k-center with weight and outliers
    millis_start: float = time.time() * 1000.
    centers, attemps, first_guess, final_guess = SeqWeightedOutliers(P=data, W=weights, k=k, z=z, alpha=alpha)
    millis_end: float = time.time() * 1000.

    # calculate time needed in milliseconds
    millis_duration: float = millis_end - millis_start

    # print output
    print("Input size n = ", data.shape[0])
    print("Number of centers k = ", k)
    print("Number of outliers z = ", z)
    print("Initial guess = ", first_guess)
    print("Final guess = ", final_guess)
    print("Number of guesses = ", attemps)
    print("Objective function = ")
    print("Time of SeqWeightedOutliers = ", millis_duration)
    print(centers)

    plt1 = plt.figure(1)
    plt.scatter(x=data[:,0], y=data[:,1], marker='.')

    plt2 = plt.figure(2)
    plt.scatter(x=centers[:,0], y=centers[:,1], marker='.')

    plt.show()





if __name__ == "__main__":
    main()