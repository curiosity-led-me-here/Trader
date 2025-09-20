import pandas as pd
import glob
from converter import convert, save_output
from process import compute_features

'''
path = "..."
examples = []

for file in glob.glob(path):
    data = convert(file)
    examples.append(data)

print(examples)
'''

path = r"/Users/ashu/Documents/Trader/NIFTY 50/2025-01-30_12-54-52.xlsx"
savepath = r'/Users/ashu/Downloads/output_.xlsx'
data = convert(path)
save_output(data, savepath)
