from array import array
import sys
import os
import numpy as np

def remove_perfect_outliers(points: np.array, weights: np.array, num_of_outiliers: int) -> np.array:
    refined_points = points.copy()
    # create matrix with all distance between all points(for each point is calculated the distance between all other points)
    n, dims = points.shape[0], points.shape[1]
    points_dist = np.zeros(shape=(n, n), dtype=points.dtype)
    for i in range(n):
        coord_diff_squared = np.square(points - points[i])
        points_dist[i] = np.sum(a=coord_diff_squared, axis=1, dtype=points.dtype)
    points_dist = points_dist[~np.eye(n,dtype=bool)].reshape(n,-1)
    print(points_dist)

    #find the points which have the maximum minimum distance between each other
    #min_dist = points_dist.min(axis=1)
    
    #outliers = min_dist[:num_of_outiliers]
    


    # now to compute the average distance of each point between all others
    #avg_dist = np.full(shape=n, fill_value=0., dtype=points.dtype)
    


    return refined_points

def main():

    # Check argv lenght and content
    assert len(sys.argv) == 4, "Usage: <Filename> <K> <Z>"

    filename = sys.argv[1]
    assert os.path.isfile(filename), "File or folder not found"
    try:
        f = open(filename)
        data = np.loadtxt(fname=filename, dtype=np.float32, delimiter=',')
    finally:
        f.close()
    
    K = sys.argv[2]
    assert K.isdigit, "K must be an integer value"
    K = int(K)
    assert K >= 2, "K must be at least 2"

    Z = sys.argv[3]
    assert Z.isdigit, "Z must be an integer value"
    Z = int(Z)
    assert Z >= 0, "K must be positive"

    np.set_printoptions(precision=2, linewidth=200)
    print(data)
    print()
    # Find outliers
    if Z > 0:
        newData = remove_perfect_outliers(data, np.zeros(shape=1), Z)
        
    
    






if __name__ == "__main__":
    main()