from pyspark import SparkContext, SparkConf
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
    return [(pair[0], id) for id in custIDs ]

            

def main():
    # CHECKING NUMBER OF CMD LINE PARAMTERS
    assert len(sys.argv) == 5, "Usage: python G001HW1_alpha.py <K> <H> <country> <file_name>"

    # SPARK SETUP
    conf = SparkConf().setAppName('G001HW1').setMaster("local[*]")
    sc = SparkContext(conf=conf)

    # INPUT READING

    # 1. Read number of partitions
    k = sys.argv[1]
    assert k.isdigit(), "K must be an integer"
    k = int(k)

    # 2. Read H(I forgot what is H)
    h = sys.argv[2]
    assert h.isdigit(), "K must be an integer"
    h = int(h)

    # 3. Read country of interest
    s = sys.argv[3]

    # 4. Read input file and subdivide it into K random partitions
    data_path = sys.argv[4]
    assert os.path.isfile(data_path), "File or folder not found"
    rawData = sc.textFile(data_path, minPartitions=k).cache()
    rawData.repartition(numPartitions=k)
    print("Number of rows =", rawData.count())

    # TASK 2: EVERYTHING BUT REMOVING DUPLICATES
    productCustomer_wDuplicates = rawData.flatMap(lambda x: checkPairsPerPartition(x,s))

    # TASK 2: REMOVING DUPLICATES (the sample 5 has only 41 pairs without duplicates)
    #productCustomer = (productCustomer_wDuplicates.groupByKey()).flatMap(removeDuplicatePairs)
    productCustomer = productCustomer_wDuplicates.distinct()
    print("Product-Customer Pairs =", productCustomer.count())
    #for r in productCustomer.collect():
    #    print(r)


    

if __name__ == "__main__":
    main()
