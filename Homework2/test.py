from ast import Lambda
from operator import pos
import time
import sys
from math import dist, sqrt
import os

from regex import R

def readVectorsSeq(filename: str):
    with open(filename) as f:
        result: list[tuple] = [tuple(map(float, i.split(','))) for i in f]
    return result

def SeqWeightedOutliers(P: list[tuple], W: list[float], k: int, z: int, alpha: float) -> list[tuple]:
    dims: int = len(P[0])
    n: int = len(P)
    attempts: int = 0

    # Compute matrix n*n containg all distances SQUARED between each and all points
    all_dist_squared: list[tuple] = []
    for i in range(n):
        i_minus_all: list[float] = []
        for j in range(n):
            dists: float = sum(tuple(map(lambda x, y: (x-y)*(x-y), P[i], P[j])))
            i_minus_all.append(dists)
        all_dist_squared.append(tuple(i_minus_all))
    
    # Compute the first guess(r) but squared
    min: float = all_dist_squared[0][1]
    for i in range(0,k+z+1):
        for j in range(i+1,k+z+1):
            if min > all_dist_squared[i][j]:
                min: float = all_dist_squared[i][j]
    r_squared: float = min / 4.
    print("Initial guess: ", sqrt(r_squared))

    while True:
        print(attempts)
        S: list[tuple] = []
        ball_radius_squared: float = (1.+2.*alpha)*(1.+2.*alpha)*r_squared  # ball radius for the center selection
        weight: list[float] = W.copy()

        for i in range(k):
            best_weight: float = 0.
            best_pt_id: int = 0

            for possible_center_id in range(n):
                current_weight: float = sum(weight[p] for p in range(n) if all_dist_squared[possible_center_id][p] < ball_radius_squared)
                if current_weight > best_weight:
                    best_weight: float = current_weight
                    best_pt_id: int = possible_center_id
            
            S.append(tuple(P[best_pt_id][j] for j in range(dims))) 
            ball_radius_squared: float = (3.+4.*alpha)*(3.+4.*alpha)*r_squared # ball radius for the covered points detection
            for j in range(n):
                if all_dist_squared[best_pt_id][j] < ball_radius_squared:
                    weight[j] = 0

        outliers_weight: float = sum(weight)
        attempts += 1

        if outliers_weight <= z:
            print("Final guess = ", sqrt(r_squared))
            print("Number of guesses = ", attempts)
            return S
        else:
            r_squared *= 4 # because it is 2^2


def ComputeObjective(inputPoints: list[tuple], solution: list[tuple], z: int) -> float :
    n: int = len(inputPoints)
    k: int = len(solution)
    min_dist_from_centers_squared: list[float] = [0.] * n
    dist_from_centers_squared: list[float] = [0.]*k

    for i in range(n):
        for j in range(k):
            dist_from_centers_squared[j] = sum(tuple(map(lambda pt, center: (pt-center)*(pt-center), inputPoints[i], solution[j])))
            min_dist_from_centers_squared[i] = min(dist_from_centers_squared)

    for i in range(z):
        min_dist_from_centers_squared.remove(max(min_dist_from_centers_squared))

    return sqrt(max(min_dist_from_centers_squared))
    

def main():
    # Check argv lenght and content
    assert len(sys.argv) == 4, "Usage: <Filename> <K> <Z>"

    filename: str = sys.argv[1]
    assert os.path.isfile(filename), "File or folder not found"
    inputPoints: list[tuple] = readVectorsSeq(filename=filename)
    
    k: str = sys.argv[2]
    assert k.isdigit(), "K must be an integer value"
    k: int = int(k)
    assert k >= 2, "K must be at least 2"

    z: str = sys.argv[3]
    assert z.isdigit(), "Z must be an integer value"
    z: int = int(z)
    assert z >= 0, "K must be positive"

    # init unit weights
    weights: list[float] = [1.] * len(inputPoints)

    # alpha
    alpha: float = 0.

    print("Input size n = ", len(inputPoints))
    print("Number of centers k = ", k)
    print("Number of outliers z = ", z)

    # run k-centers with outliers
    millis_start: float = time.time() * 1000.
    solution: list[tuple] = SeqWeightedOutliers(P=inputPoints, W=weights, k=k, z=z, alpha=alpha)
    millis_end: float = time.time() * 1000.

    # calculate time needed in milliseconds
    millis_duration: float = millis_end - millis_start

    # Compute objective
    obj: float = ComputeObjective(inputPoints=inputPoints, solution=solution, z=z)

    #print result
    print("Objective function = ", obj)
    print("Time of SeqWeightedOutliers = ", millis_duration)
    print(solution)

if __name__ == "__main__":
    main()