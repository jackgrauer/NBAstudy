import pandas as pd
from code.CENTRALraw import CENTRAL

df = pd.DataFrame(CENTRAL)

def process_CENTRAL():
    # Initialize DataFrame with provided data
    df = pd.DataFrame(CENTRAL)