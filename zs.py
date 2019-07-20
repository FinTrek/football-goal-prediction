"""

Atheesh Krishnan

"""
import pandas as pd
import pandas_profiling

df = pd.read_csv("D:/DS/zs_data.csv")
pandas_profiling.ProfileReport(df)