#!/usr/bin/python

# reducer for hadoop scripts that take in output from libsvm, a popular support vector machine software program,
# trained on a subset of the data, and finds the class for a larger data set
# all of the work was done by the mapper, so the reducer just copies mapper output

import sys

# a line of mapper output looks like:
# 1,class1
# 2,class2
# 3,class2,...
# the first number in each line is a label/key for the data point
# the second is a label for the class, class1 or class2
def read_mapper_output(file):
    for line in file:
        yield line.strip()

def main():
    data = read_mapper_output(sys.stdin)
    for datum in data:
        print datum
        
if __name__ == "__main__":
    main()
