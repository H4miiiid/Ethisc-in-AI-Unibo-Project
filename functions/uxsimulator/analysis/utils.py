import pandas as pd
import re


def read_output(
    namefile: str, 
    datapath: str = "data/results",
    namecols: list = None,
    dtype = None
) -> pd.DataFrame:
    df = pd.read_csv(f"{datapath}/{namefile}", header=None if namecols else 'infer', dtype=dtype)
    if namecols:
        df.columns = namecols
    return df


def is_valid_string(s: str) -> bool:
    pattern = r'^[0-9_]+$'
    return bool(re.match(pattern, s))