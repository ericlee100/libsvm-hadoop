#!/usr/bin/python

# run libsvm, a support vector machine program, on a subset of your data
# script you're reading is a python hadoop mapper script that takes in output from libsvm
# then predicts class on a larger input data set
# currently works only for 2 class problems using any kernel except for pre-computed: rbf, linear, sigmoid, polynomial
# results will be identical to libsvm for linear, sigmoid and polynomial
# results may differ for rbf when features are present in the testing data that are completely missing from the training data

import numpy as np
import sys
import urllib2

# boolean flag indicating whether script is intended for Amazon elastic map reduce or not
flagAmazonMadReduce = False

# files libsvm generates during the training process that contain the range of the feature values, support vectors and other parameters
rangeFile = "/home/user/amazonHadoop/config/train.range"
modelFile = "/home/user/amazonHadoop/config/train.model"
# if you're using Amazon EMR, then I assume that you'll put the libsvm range and model files in Amazon data storage, S3, and make public the folder
# containing the files
#rangeFile = "https://s3.amazonaws.com/bucketName/config/a2aTrain.range"
#modelFile = "https://s3.amazonaws.com/bucketName/config/a2aTrain.model"

# libsvm produces a range file after you scale the data
# format:
#   x           features
#   -1 1        min/max of all scaled features
#   1 -22 45    min/max of unscaled feature # 1
#   2 ...       feature 2
# create arrays of slope and intercept to convert unscaled variables to scaled
def readRangeFile(rangeFileName):
    # if you're running on Amazon map reduce, I assume that the range file is on their S3 storage device, which is easiest to access through https
    # and needs to be opened differently
    if flagAmazonMadReduce==False:
        # get maximum feature index, numFeatures
        with open(rangeFileName, 'r') as f:
            # get the largest feature number listed
            numFeatures = 0
            for line1 in f:
                try:
                    i = int(line1.strip().split(' ')[0])
                except ValueError:
                    i = 0
                numFeatures = max([i,numFeatures])
                
        slopeArr = np.zeros(numFeatures)
        interceptArr = np.zeros(numFeatures)

        with open(rangeFileName, 'r') as f:
            linesSinceDefiningType = 0
            for line1 in f:
                if linesSinceDefiningType>0:
                    lineParse = [float(x) for x in line1.split( )]
                    if(linesSinceDefiningType==1):
                        scaledMinMax = lineParse[:2]
                    else:
                        featureIndx = int(lineParse[0]) - 1
                        unscaledMinMax = lineParse[1:]
                        slopeArr[featureIndx] = (scaledMinMax[1] - scaledMinMax[0]) / (unscaledMinMax[1] - unscaledMinMax[0])
                        interceptArr[featureIndx] = ( scaledMinMax[1] - slopeArr[featureIndx]*unscaledMinMax[1] )
                linesSinceDefiningType += 1
    else:
        f = urllib2.urlopen(rangeFileName)
        # count lines in file and create arrays
        numFeatures = 0
        for line1 in f.readlines():
            try:
                i = int(line1.strip().split(' ')[0])
            except ValueError:
                i = 0
            numFeatures = max([i,numFeatures])
        f.close()    

        slopeArr = np.zeros(numFeatures)
        interceptArr = np.zeros(numFeatures)

        linesSinceDefiningType = 0
        f = urllib2.urlopen(rangeFileName)
        for line1 in f.readlines():
            if linesSinceDefiningType>0:
                lineParse = [float(x) for x in line1.split( )]
                if(linesSinceDefiningType==1):
                    scaledMinMax = lineParse[:2]
                else:
                    featureIndx = int(lineParse[0]) - 1
                    unscaledMinMax = lineParse[1:]
                    slopeArr[featureIndx] = (scaledMinMax[1] - scaledMinMax[0]) / (unscaledMinMax[1] - unscaledMinMax[0])
                    interceptArr[featureIndx] = ( scaledMinMax[1] - slopeArr[featureIndx]*unscaledMinMax[1] )                        
            linesSinceDefiningType += 1
        f.close()
    return [slopeArr,interceptArr]

# read libsvm's model file
# model file starts with info like the kernel type, number of support vectors, etc., then lists support vectors
# featureCount is the number of features you get from the range file
def readModelFile(fileName,featureCount):
    # if you're running on Amazon map reduce, I assume that the range file is on their S3 storage device, which is easiest to access through https  
    # and needs to be opened differently
    if flagAmazonMadReduce==False:
        with open(fileName ,'r') as f:
            # flag that says when support vectors are being listed
            flagSV = False
            # support vector counter
            svCount = 0
            for line1 in f:
                line1 = (line1.strip()).split()
                if flagSV==False:
                    if line1[0]!="SV":
                        if line1[0]=="svm_type":
                            svm_type=line1[1]
                        if line1[0]=="kernel_type":
                            kernel_type=line1[1]
                        if line1[0]=="gamma":
                            gamma = float(line1[1])
                        if line1[0]=="nr_class":
                            nr_class=int(line1[1])
                        if line1[0]=="total_sv":
                            total_sv=int(line1[1])
                            # create array for alpha_i*y_i value associated with each support vector where alpha_i is the support vector's weight
                            alphaYArr = np.zeros((total_sv))
                            # define a matrix conaining support vectors
                            svMatrix = np.zeros( (total_sv,featureCount) )
                        if line1[0]=="rho":
                            rho=-float(line1[1])
                        if line1[0]=='label':
                            label=line1[1:]
                        if line1[0]=='nr_sv':
                            nr_sv=[int(ii) for ii in line1[1:]]
                        if line1[0]=='degree':
                            degree=int(line1[1])
                        if line1[0]=='coef0':
                            coef0=float(line1[1])
                    else:
                        flagSV = True
                else:
                    #SV example - 0.5 1:0.534744 3:-0.333333 4:-0.333333 5:-0.111111 6:0.333333 7:1 8:-0.555556 9:-0.777778 10:-1
                    #             SV weight*y_1, SV values for features 1, 3, etc.
                    # Note in example that features with value = 0 are missing.
                    # If you trace through libsvm c/c++ code, when a feature in a data point is missing, it's treated as 0, not the mean/median
                    alphaYArr[svCount] = float(line1[0])
                    indx1 = [int(ii.split(':')[0])-1 for ii in line1[1:]]
                    val1 = [float(ii.split(':')[1]) for ii in line1[1:]]                    
                    svMatrix[svCount,indx1] = val1
                    # remember that you want weight * y_i * K(x*x')
                    
                    svCount += 1
    else:
        f = urllib2.urlopen(fileName)
        # flag that says when support vectors are being listed
        flagSV = False
        # support vector counter
        svCount = 0
        for line1 in f.readlines():
            line1 = (line1.strip()).split()
            if flagSV==False:
                if line1[0]!="SV":
                    if line1[0]=="svm_type":
                        svm_type=line1[1]
                    if line1[0]=="kernel_type":
                        kernel_type=line1[1]
                    if line1[0]=="gamma":
                        gamma = float(line1[1])
                    if line1[0]=="nr_class":
                        nr_class=int(line1[1])
                    if line1[0]=="total_sv":
                        total_sv=int(line1[1])
                        # crete array for alpha_i*y_i value associated with each support vector where alpha_i is the support vector's weight
                        alphaYArr = np.zeros((total_sv))
                        # define a matrix conaining support vectors
                        svMatrix = np.zeros( (total_sv,featureCount) )
                    if line1[0]=="rho":
                        rho=-float(line1[1])
                    if line1[0]=='label':
                        label=line1[1:]
                    if line1[0]=='nr_sv':
                        nr_sv=[int(ii) for ii in line1[1:]]
                    if line1[0]=='degree':
                        degree=int(line1[1])
                    if line1[0]=='coef0':
                        coef0=float(line1[1])
                else:
                    flagSV = True
            else:
                #SV example - 0.5 1:0.534744 3:-0.333333 4:-0.333333 5:-0.111111 6:0.333333 7:1 8:-0.555556 9:-0.777778 10:-1
                #             SV weight*y_i, SV values for feature 1, 2, etc.
                alphaYArr[svCount] = float(line1[0])
                indx1 = [int(ii.split(':')[0])-1 for ii in line1[1:]]
                val1 = [float(ii.split(':')[1]) for ii in line1[1:]]                    
                svMatrix[svCount,indx1] = val1
                
                svCount += 1
        f.close()

    if kernel_type=="rbf":       
        returnVal = [alphaYArr,svMatrix, kernel_type,label,gamma,None,None,rho]
    if kernel_type=="linear":       
        returnVal = [alphaYArr,svMatrix, kernel_type,label,None,None,None,rho]
    if kernel_type=="polynomial":       
        returnVal = [alphaYArr,svMatrix, kernel_type,label,gamma,coef0,degree,rho]
    if kernel_type=="sigmoid":       
        returnVal = [alphaYArr,svMatrix, kernel_type,label,gamma,coef0,None,rho]

    return( returnVal )


# sum( alpha_i * y_i * Kernel(x,x_i) ), a term used in the predictor function
# kernel types: linear, polynomial, sigmoid rbf(radial basis function)
def sumAlphaYKernel(kernel_type,svMatrix,x,alphaY,gamma=None,coef0=None,degree=None):
    if kernel_type=="rbf":
        diffVec = svMatrix - x
        kernel1 = np.exp(-gamma * np.sum(diffVec**2,axis=-1))
    if kernel_type=="linear":
        kernel1 = np.dot(svMatrix,x)
    if kernel_type=="polynomial":
        kernel1 = ( gamma*np.dot(svMatrix,x) + coef0 )**degree
    if kernel_type=="sigmoid":
        kernel1 = np.tanh(gamma * np.dot(svMatrix,x) + coef0)

    return( np.dot(alphaY,kernel1) )

# return highVal if value1 is above cutoff1, lowVal otherwise
def plusMinus(value1,cutoff1,highVal,lowVal):
    if(value1>=cutoff1):
        return highVal
    else:
        return lowVal

# get the class of one point
def predictOnePoint(x,beta0,alphaY,svMatrix,kernel_type,categoryLabels,gamma=None,coef0=None,degree=None):
    # the standard SVM class prediction function for the dual Lagrangian
    val1 = beta0 + sumAlphaYKernel(kernel_type=kernel_type,svMatrix=svMatrix,x=x,alphaY=alphaY,gamma=gamma,coef0=coef0,degree=degree)
    return( plusMinus(value1=val1,cutoff1=0,highVal=categoryLabels[0],lowVal=categoryLabels[1]) )

# generator that reads in lines of data
def read_input(file,separator=','):
    for line1 in file:
        yield line1

def main(separator=','):
    range1=readRangeFile(rangeFile)
    svList=readModelFile(fileName=modelFile,featureCount=len(range1[0]))

    data = read_input(sys.stdin)
    #data = open("/home/abc/temp/amazonHadoop/input/a2aTestShortShortNum",'r')
    # data format: 001 1:857774.000000 3:1.000000 4:1.000000 5:1.000000 6:3.000000 7:1.000000 8:2.000000 9:2.000000 10:1.000000\n
    # format is the same as standard libsvm, but the first value, 001, is the key in the key,value pair for map reduce
    # other values are feature number:feature weight
    for datum in data:
        # scale predictors
        keyVal = datum.strip().split(' ')
        # remove features in the test data that aren't present in the training data        
        featureNoKey = filter(lambda x: int(x.split(':')[0])<=len(range1[0]), keyVal[1:])

        indx1 = [int(ii.split(':')[0])-1 for ii in featureNoKey]
        val1 = [float(ii.split(':')[1]) for ii in featureNoKey]
        # the input data point you read in may not contain values for all features - fill in missing values with zeroes
        fullVal = np.zeros(len(range1[0]))
        fullVal[indx1] = val1

        # scale the data point
        fullVal = fullVal*range1[0] + range1[1]
        oneVal = predictOnePoint(x=fullVal,beta0=svList[7],alphaY=svList[0],svMatrix=svList[1],kernel_type=svList[2],categoryLabels=svList[3],gamma=svList[4],
                                 coef0=svList[5],degree=svList[6])
        print '%s%s%s' % (keyVal[0], separator, oneVal)

if __name__ == "__main__":
    main()
