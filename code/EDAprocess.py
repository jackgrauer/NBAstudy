import pandas as pd
from code.EDAraw import EDA

def process_EDA():
     # Step 1: Initialize DataFrame with provided data
    df = pd.DataFrame(EDA)

    print (df)
    return df

