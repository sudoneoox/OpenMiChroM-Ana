import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import normalize


""" Analysis.py function definitions """
def getHiCData_simulation(filepath):
    """
    returns
    r: hic data
    D: scaling
    err: error
    """
    contactMap = np.loadtxt(filepath)
    r=np.triu(contactMap, k=1) 
    r = normalize(r, axis=1, norm='max') 
    rd = np.transpose(r) 
    r=r+rd + np.diag(np.ones(len(r))) 

    D1=[]
    err = []
    for i in range(0,np.shape(r)[0]): 
        D1.append((np.mean(np.diag(r,k=i)))) 
        err.append((np.std(np.diag(r,k=i))))
    
    return(r,D,err)

def getHiCData_experiment(filepath, cutoff=0.0, norm="max"):
        """
        returns
        r: hic data
        D: scaling
        err: error
        """
        contactMap = np.loadtxt(filepath)
        r = np.triu(contactMap, k=1)
        r[np.isnan(r)]= 0.0
        r = normalize(r, axis=1, norm="max")
        
        if norm == "first":
            for i in range(len(r) - 1):
                maxElem = r[i][i + 1]
                if(maxElem != np.max(r[i])):
                    for j in range(len(r[i])):
                        if maxElem != 0.0:
                            r[i][j] = float(r[i][j] / maxElem)
                        else:
                            r[i][j] = 0.0 
                        if r[i][j] > 1.0:
                            r[i][j] = .5
        r[r<cutoff] = 0.0
        rd = np.transpose(r) 
        r=r+rd + np.diag(np.ones(len(r)))
    
        D1=[]
        err = []
        for i in range(0,np.shape(r)[0]): 
            D1.append((np.mean(np.diag(r,k=i)))) 
            err.append((np.std(np.diag(r,k=i))))
        D=np.array(D1)#/np.max(D1)
        err = np.array(err)
    
        return(r,D,err)