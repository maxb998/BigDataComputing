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

    if ((Quantity > 0) & (s == "all")) | ((Quantity > 0) & (s == Country)):
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
    assert H.isdigit(), "K must be an integer"
    H = int(H)

    # 3. Read country of interest
    S = sys.argv[3]

    # 4. Read input file and subdivide it into K random partitions
    data_path = sys.argv[4]
    assert os.path.isfile(data_path), "File or folder not found"
    rawData = sc.textFile(data_path, minPartitions=K).cache()
    rawData.repartition(numPartitions=K)
    print("Number of rows =", rawData.count())

    # TASK 2: EVERYTHING BUT REMOVING DUPLICATES
    productCustomer_wDuplicates = rawData.flatMap(lambda x: checkPairsPerPartition(x,S))

    # TASK 2: REMOVING DUPLICATES (the sample 5 has only 41 pairs without duplicates)
    productCustomer = (productCustomer_wDuplicates.groupByKey()).flatMap(removeDuplicatePairs)
    #productCustomer = productCustomer_wDuplicates.distinct()
    print("Product-Customer Pairs =", productCustomer.count())
    #for r in productCustomer.collect():
    #    print(r)

    # TASK 3:
    productPopularity1 = productCustomer.mapPartitions(lambda x: x).groupByKey().mapValues(len)
    '''
    for prod in productPopularity1.collect():
        print(prod)'''
    
    # TASK 4:
    productPopularity2 = productCustomer.map(lambda x: (x[0],1)).reduceByKey(add)
    '''
    for prod in productPopularity2.collect():
        print(prod)'''
    
    # TASK 5:
    if H>0:
        most_popular_H = productPopularity1.sortBy(lambda x: x[1], ascending=False).take(H)
        print(f"Top {H} Products and their Popularities:")
        for pop in most_popular_H:
            print(f"Product {pop[0]}, Popularity {pop[1]}; ")
        
    # TASK 6:
    if H==0:
        most_popular1 = productPopularity1.sortByKey().collect()
        print("\n productPopularity1:")
        for pop in most_popular1:
            print(f"Product {pop[0]}, Popularity {pop[1]}; ")
        most_popular2 = productPopularity2.sortByKey().collect()
        print("\n productPopularity2:")
        for pop in most_popular2:
            print(f"Product {pop[0]}, Popularity {pop[1]}; ")

    

if __name__ == "__main__":
    main()
