import pandas as pd
import numpy as np 
def getdata():
	preA = np.array(pd.read_csv("DataPredict_complete.csv",header=None,sep=','))
	preA = preA[1:2**13+1,:-2]
	A = preA.astype(np.float64)
	return A
