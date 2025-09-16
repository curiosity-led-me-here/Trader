import numpy as np
import pandas as pd
from converter import convert, split

path = r'/Users/ashu/Documents/Trader/data/NIFTY 51/2024-11-28_16-38-19.xlsx'

df = convert(path)
df = split(df)

df = pd.DataFrame(df)
print(df)