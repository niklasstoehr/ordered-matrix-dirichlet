import numpy as np
import pandas as pd
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import torch
from matplotlib.pyplot import cm
import matplotlib as mpl

from omd0configs import configs

mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False


def set_random_seed(random_seed):

    if random_seed == None:
        random_seed = torch.randint(0, 1000000, (1,)).item()
    return random_seed


def store_data(data = None, path: str = "", file: str = "", file_end: str = ".pkl"):

    config = configs.ConfigBase()
    path_name = config.get_path(path) / (str(file) + str(file_end))
    with open(path_name, 'wb') as f:
        pickle.dump(data, f)
    print(f"stored to {path_name}")


def load_data(path: str = "", file: str = "", file_end: str = ".pkl", if_fail ={}):

    config = configs.ConfigBase()
    path_name = config.get_path(path) / (str(file) + str(file_end))
    try:
        with open(path_name, 'rb') as f:
            data = pickle.load(f)
        print(f"loaded from {path_name}")
    except:
        print(f"no file found at {path_name}")
        data = if_fail["fn"](*if_fail["args"])
    return data


def store_csv(df=None, path_name = "latents", file_name = "df"):

    config = configs.ConfigBase()
    path_name = config.get_path(path_name) / Path(file_name + ".csv")
    df.to_csv(path_name, index=False)
    print(f"stored csv to {path_name, file_name}")


def load_csv(path_name = "latents", file_name="df", if_fail ={}):

    config = configs.ConfigBase()
    path_name = config.get_path(path_name) / Path(str(file_name) + ".csv")
    try:
        csv_data = pd.read_csv(path_name)
        print(f"loaded csv from {path_name, file_name}")
    except:
        print(f"no file found at {path_name}")
        try:
            csv_data = if_fail["fn"](*if_fail["args"])
        except:
            csv_data = None
    return csv_data

