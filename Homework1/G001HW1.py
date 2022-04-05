from pyspark import SparkContext, SparkConf
from operator import add
import sys
import os
import random as rand

def checkPairsPerPartition(transaction, s):

    arg = transaction.split(',')
    prodID = arg[1]
    CustID = int(arg[6])
    Quantity = int(arg[3])
    Country = arg[7]

    if (Quantity > 0 and ((s == "all") or(s == Country))):
        return [(prodID, CustID)]
    else :
        return []

def removeDuplicatePairs(pair):

    custIDs = []
    for custID in pair[1]:
        if custID not in custIDs:
            custIDs.append(custID)
    return [(pair[0], i) for i in custIDs ]

            
def main():
    # CHECKING NUMBER OF CMD LINE PARAMETERS
    assert len(sys.argv) == 5, "Usage: python G001HW1.py <K> <H> <S> <file_name>"

    # SPARK SETUP
    conf = SparkConf().setAppName('G001HW1').setMaster("local[*]")
    sc = SparkContext(conf=conf)

    # INPUT READING

    # 1. Read number of partitions
    K = sys.argv[1]
    assert K.isdigit(), "K must be an integer"
    K = int(K)

    # 2. Read H
    H = sys.argv[2]
    assert H.isdigit(), "H must be an integer"
    H = int(H)

    # 3. Read country of interest 
    S = sys.argv[3]
    
    # 4. Read input file 
    data_path = sys.argv[4]
    assert os.path.isfile(data_path), "File or folder not found"

    # TASK 1: Subdivide into K partitions the RDD of strings rawData created by reading the input file 
    rawData = sc.textFile(data_path, minPartitions=K).cache()
    rawData.repartition(numPartitions=K)
    # Then print the number of rows read from the input file
    print("Number of rows =", rawData.count())

    # TASK 2: Transform rawData into an RDD of (ProductID,CostumerID) pairs called productCustomer
    productCustomer_wDuplicates = rawData.flatMap(lambda x: checkPairsPerPartition(x,S))
    # Remove duplicates
    productCustomer = (productCustomer_wDuplicates.groupByKey()).flatMap(removeDuplicatePairs)
    print("Product-Customer Pairs =", productCustomer.count())

    # TASK 3: Transform productCustomer into an RDD of (ProductID,Popularity) pairs called productPopularity1 using mapPartitions
    # Popularity is the number of distinct customers from Country S that purchased a positive quantity of product ProductID
    productPopularity1 = productCustomer.mapPartitions(lambda x: x).groupByKey().mapValues(len)
    
    # TASK 4: Transform productCustomer into an RDD of (ProductID,Popularity) pairs called productPopularity2 using map
    productPopularity2 = productCustomer.map(lambda x: (x[0],1)).reduceByKey(add)
    
    # TASK 5: Print the ProductID and Popularity of the H>0 products with highest Popularity
    if H>0:
        most_popular_H = productPopularity1.sortBy(lambda x: x[1], ascending=False).take(H)
        print(f"Top {H} Products and their Popularities:")
        for pop in most_popular_H:
            print(f"Product {pop[0]} Popularity {pop[1]};", end = " ")
        
    # TASK 6: If H=0, collect and print all pairs of productPopularity1\productPopularity2 in increasing lexicographic order of ProductID
    if H==0:
        most_popular1 = productPopularity1.sortByKey().collect()
        print("\nproductPopularity1:")
        for pop in most_popular1:
            print(f"Product: {pop[0]} Popularity: {pop[1]};", end = " ")
        most_popular2 = productPopularity2.sortByKey().collect()
        print("\nproductPopularity2:")
        for pop in most_popular2:
            print(f"Product: {pop[0]} Popularity: {pop[1]};", end = " ")
    print("\n")

    

if __name__ == "__main__":
    main()
