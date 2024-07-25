
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import itertools
from collections import Counter
pd.options.mode.chained_assignment = None  # default='warn'
tqdm.pandas()

from omd1data.conflict import icews_data


class CountBuilder():
    """
    return S x T x A x N count tensor
    """

    def __init__(self, df, time_level="M", action_level="G_A_rank", min_V_count=1000, tensor_shape = 4):
        required_columns = ["V1_name", "V2_name", "date"] + [action_level]
        section = list(set(required_columns).intersection(list(df.columns)))
        assert len(section) == 4

        self.tensor_shape = tensor_shape
        df = self.filter_by_V(df, min_V_count=min_V_count)
        df, self.mappings, self.inverse_mappings = self.select_columns(df, action_level=action_level, time_level=time_level)
        self.df_counts = self.build_counts(df, self.mappings)
        self.data = self.build_tensor(self.df_counts, self.mappings)

    def filter_by_V_pairs(self, df, min_V_count=100):
        df_ST = df.groupby(["V1_name", "V2_name"]).size().reset_index(name='counts')
        df_ST = df_ST.sort_values(by=['counts']).reset_index(drop=True)
        df_ST = df_ST[df_ST["counts"] >= min_V_count]
        i_ST = df_ST.set_index(["V1_name", "V2_name"]).index
        i_df = df.set_index(["V1_name", "V2_name"]).index
        df = df[i_df.isin(i_ST)].reset_index(drop=True)
        assert df.groupby(['V1_name', 'V2_name']).size().reset_index(name='counts')["counts"].min() >= min_V_count
        return df

    def filter_by_V(self, df, min_V_count=1000, omit_self_action = True):
        ## filter self-targeted actions
        if omit_self_action :
            df = df[df['V1_name'] != df['V2_name']]

        ## filter quantity
        actors = sorted(df["V1_name"].to_list() + df["V2_name"].to_list())
        actor_counter = Counter(actors)
        V_list = [actor for actor, count in actor_counter.items() if count >= min_V_count]
        df = df[df['V1_name'].isin(V_list)]
        df = df[df['V2_name'].isin(V_list)]
        return df

    def select_columns(self, df, action_level="A_rank", Afriend2enemy=True, time_level="M"):
        ## date T
        df.loc[:, "date"] = df["date"].dt.to_period(time_level).copy(deep=True)
        date_value_list = sorted(list(df["date"].unique()))
        date_list = np.arange(0, len(date_value_list), 1)
        date_dict = dict(zip(date_value_list, date_list))
        date_dict_inv = {v: k for k, v in date_dict.items()}
        df['T'] = df['date'].map(date_dict).astype('int32')

        ## actors V
        V_value_list = sorted(list(set(df["V1_name"].to_list() + df["V2_name"].to_list())))
        V_list = np.arange(0, len(V_value_list), 1)
        V_dict = dict(zip(V_value_list, V_list))
        V_dict_inv = {v: k for k, v in V_dict.items()}
        df['V1'] = df['V1_name'].map(V_dict).astype('int32')
        df['V2'] = df['V2_name'].map(V_dict).astype('int32')

        ## A
        df_sort = df.sort_values(by=[action_level]).groupby(action_level).first().reset_index()
        A_value_list = df_sort[action_level].to_list()
        A_name_list = df_sort[action_level.split("_")[1] + "_name"].to_list()
        A_G_list = df_sort["G_" + action_level.split("_")[1]].to_list()
        A_list = np.arange(0, len(A_value_list), 1).astype('int32')
        if Afriend2enemy: ## make action type 0 friendly and 19 hostily
            A_list = A_list[::-1]
        A_dict = dict(zip(A_value_list, A_list))
        A_dict_inv = {v: k for k, v in sorted(A_dict.items())}
        A_name_dict_inv = dict(sorted(zip(A_list, A_name_list)))
        A_G_dict_inv = dict(sorted(zip(A_list, A_G_list)))
        df['A'] = df[action_level].map(A_dict).astype('int32')


        df = df[["T", "V1", "V2", "A"]]
        mappings = {"T": date_dict, "V1": V_dict, "V2": V_dict, "A": A_dict}
        inverse_mappings = {"T": date_dict_inv, "V1": V_dict_inv, "V2": V_dict_inv, "A": A_dict_inv, "A_name": A_name_dict_inv, "A_G": A_G_dict_inv}
        return df, mappings, inverse_mappings


    def build_counts(self, df, mappings):
        combined = list(list(v.values()) for v in mappings.values())
        base_df = pd.DataFrame(columns=['T', 'V1', 'V2', 'A'], data=list(itertools.product(*combined)))
        df_all = pd.concat([df, base_df], ignore_index=True)
        df_all = df_all.groupby(['T', 'V1', 'V2', 'A']).size().reset_index(name='counts')
        assert len(df_all) == len(base_df)
        df_all["counts"] = df_all["counts"] - 1
        return df_all

    def build_tensor(self, df, mappings):
        ## V1 x V2 x A x T
        V = len(mappings["V1"])
        A = len(mappings["A"])
        T = len(mappings["T"])

        data = np.zeros((V, V, A, T))

        for t in range(0, T):
            df_slice = df[df["T"] == t]
            # indices = df_slice[["V1", "V2", "A", "T"]].to_numpy()
            # indices = np.reshape(indices, (ST, ST, A, 4))
            values = df_slice[["counts"]].to_numpy().squeeze()
            values = np.reshape(values, (V, V, A))
            data[..., t] = values

        data = torch.FloatTensor(data)
        if self.tensor_shape == 3:
            data = data.view(-1, data.shape[-2], data.shape[-1])
        return data



if __name__ == '__main__':
    icews = icews_data.ICEWS(file_list=[2017], V1=[], V2=[], start_end=["2000-01-01", "2022-01-01"])
    cb = CountBuilder(icews.df, time_level="M", action_level="G_A_rank", min_V_count=5000, tensor_shape = 4)
    print(cb.data.shape)
    print(cb.inverse_mappings["A_G"])