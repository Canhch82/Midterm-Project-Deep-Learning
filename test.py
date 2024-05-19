'''
This file contains the tests for evaluating the functions in nn.py
'''

import sklearn.datasets as datasets
import nn
import numpy as np
from optimizer import optimizer

def test_run():
    """
    Sample test run.

    :return: None
    """
    
    # test run for binary classification problem:
    np.random.seed(3)
    print('Running a binary classification test')

    #Generate sample binary classification data
    data = datasets.make_classification(n_samples=30000,n_features=10,n_classes=2)
    X= data[0].T
    Y = (data[1].reshape(30000,1)).T
    net = nn.nn([10,20,1],['relu','sigmoid'])
    net.cost_function = 'CrossEntropyLoss'
    optim = optimizer.SGDOptimizer
    optim(X,Y,net,128,alpha=0.07,epoch=5,lamb=0.05,print_at=1,momentum=0.9)
    output = net.forward(X)
    output = 1*(output>=0.5)
    accuracy = np.sum(output==Y)/30000
    print('for stochaistic gradient descenet with momentum\n accuracy = ' ,accuracy*100)

    
   

if __name__ == "__main__":
    test_run()

