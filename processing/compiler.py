import pandas as pd
import glob
from converter import convert
from process import compute_features

path = "..."
examples = []

for file in glob.glob(path):
    data = convert(file)
    data = compute_features(data)
    examples.append(data)

print(examples)