from array import array
import sys
import os
import numpy as np

def findPerfectOutliers(points: np.array, num_of_outiliers: int) -> np.array:
    refined_points = points.copy()
    # create matrix with all distance between all points(for each point is calculated the distance between all other points)
    n, dims = points.shape[0], points.shape[1]
    points_dist = np.zeros(shape=(n, n), dtype=points.dtype)
    onesMat = np.full(shape=())
    for i in range(n):
        
        points_dist[i] = points - points[i]
    
    # now to compute the average distance of each point between all others
    avgDist = np.full(shape=n, fill_value=1., dtype=points.dtype)
    for i in range(n):


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


    # Find outliers
    if Z > 0:
        a = 0
        
    
    






if __name__ == "__main__":
    main()