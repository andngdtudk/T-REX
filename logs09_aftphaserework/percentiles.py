import numpy as np, pandas as pd
df = pd.read_csv('logs/delta_log.csv', names=['id','delta'])
print(df['delta'].describe(percentiles=[.05,.25,.75,.95,.99]))