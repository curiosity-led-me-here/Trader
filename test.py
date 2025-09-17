import pandas as pd
import numpy as np

'''
path = r'/Users/ashu/Documents/Trader/NIFTY 50/2025-04-24_13-03-28.xlsx'
df = pd.read_excel(path)

print(df["dateTime"].iloc[-1])
'''

array1 = np.array([91.55,54.36,27.65,12.55])
array2 = np.array([14.7,28.2,51.2,85.6])

grad1 = np.gradient(array1, array2)
print(grad1)

grad2 = (array1[1] - array1[0])/(array2[1] - array2[0])
grad3 = np.gradient(array1, grad2)
print(grad3)