import pandas as pd
import glob
from converter import convert

path = r"/Users/ashu/Documents/Trader/data/NIFTY 51/*.xlsx"
examples = []

for file in glob.glob(path):
    data = convert(file)
    examples.append(data)