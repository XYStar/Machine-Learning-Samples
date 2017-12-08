# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 15:38:16 2017

@author: Jesse Ronald Estoque
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

#print(cancer.DESCR) # Print the data set description
print(cancer.keys())

#How many features does the breast cancer dataset have?
def answer_zero():
    # This function returns the number of features of the breast cancer dataset, which is an integer. 
    # The assignment question description will tell you the general format the autograder is expecting
    return len(cancer['feature_names'])

# You can examine what your function returns by calling it in the cell. If you have questions
# about the assignment formats, check out the discussion forums for any FAQs
print(answer_zero())

'''
Question 1
Scikit-learn works with lists, numpy arrays, scipy-sparse matrices, and pandas DataFrames, so converting the dataset to a DataFrame is not necessary for training this model. Using a DataFrame does however help make many things easier such as munging data, so let's practice creating a classifier with a pandas DataFrame.
Convert the sklearn.dataset cancer to a DataFrame.
This function should return a (569, 31) DataFrame with
columns =
['mean radius', 'mean texture', 'mean perimeter', 'mean area',
'mean smoothness', 'mean compactness', 'mean concavity',
'mean concave points', 'mean symmetry', 'mean fractal dimension',
'radius error', 'texture error', 'perimeter error', 'area error',
'smoothness error', 'compactness error', 'concavity error',
'concave points error', 'symmetry error', 'fractal dimension error',
'worst radius', 'worst texture', 'worst perimeter', 'worst area',
'worst smoothness', 'worst compactness', 'worst concavity',
'worst concave points', 'worst symmetry', 'worst fractal dimension',
'target']
and index =
RangeIndex(start=0, stop=569, step=1)
'''
def answer_one():
    
    # Your code here
    df = pd.DataFrame(cancer.data,columns=cancer.feature_names)
    df['target']=cancer.target
    return df # Return your answer


print(answer_one())

'''
Question 2
What is the class distribution? (i.e. how many instances of malignant (encoded 0) and how many benign (encoded 1)?)
This function should return a Series named target of length 2 with integer values and index = ['malignant', 'benign']
'''
def answer_two():
    cancerdf = answer_one()
    
    # Your code here
    
    return # Return your answer


answer_two() 