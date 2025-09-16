import pandas as pd
import glob
from ancilliaries.converter import convert
from ancilliaries.process import compute_features

path = r"/Users/ashu/Documents/Trader/data/NIFTY 51/*.xlsx"
examples = []

for file in glob.glob(path):
    data = convert(file)
    data = compute_features(data)
    examples.append(data)

print(examples)