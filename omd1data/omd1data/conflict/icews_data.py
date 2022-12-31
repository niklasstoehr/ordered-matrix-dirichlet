import pandas as pd
import os
from tqdm import tqdm
import numpy as np
tqdm.pandas()

from omd0configs import configs

class ICEWS():

    def __init__(self, file_list=[2020], V1=[], V2=[], start_end=["2000-01-01", "2022-01-01"]):

        self.config = configs.ConfigBase()
        self.file_list = file_list
        df = self.load_raw_data(file_list)
        df = self.filter_columns(df)
        df = self.prepare_action_types(df)
        df = self.prepare_source_target(df, V1, V2)
        self.df = self.prepare_date(df, start_end)

    def load_raw_data(self, file_list=[2020]):

        file_paths = []
        config = configs.ConfigBase()
        files = os.listdir(config.get_path("icews"))

        df_list = []
        for f in files:
            if "events" in f.lower(): ## avoid e.g. .ds_store
                f_arr = f.split(".")
                if int(f_arr[1]) in file_list or len(file_list) == 0:
                    file_paths.append(config.get_path("icews") / f)
                    file_df = pd.read_csv(file_paths[-1], sep="\t", converters={"CAMEO Code":self.convert_cameo})
                    df_list.append(file_df)
        df = pd.concat(df_list, axis=0, ignore_index=True)
        del df_list
        return df


    def convert_cameo(selfs, cameo_code):
        try:
            cameo_code = float(cameo_code)
        except:
            if cameo_code == '13y': ## problem in icews 2017 data
                cameo_code = 13.0
            else:
                cameo_code = np.nan
        return cameo_code


    def filter_columns(self, df):

        rename_dict = {"Event ID": "id", "Event Date": "date", "Source Country": "V1_name", "Target Country": "V2_name",
                       "CAMEO Code": "E_code"}
        df = df[df.columns.intersection(list(rename_dict.keys()))]
        df = df.rename(columns=rename_dict)
        df = df.dropna(inplace=False).reset_index(drop=True)
        country_map = {"Russian Federation": "Russia", "United States": "USA", "United Kingdom": "UK", "Occupied Palestinian Territory": "Palestine", "United Arab Emirates": "UAE"}
        df["V1_name"] = df["V1_name"].replace(country_map)
        df["V2_name"] = df["V2_name"].replace(country_map)
        return df


    def prepare_action_types(self, df):

        goldstein_map_df = pd.read_csv(self.config.get_path("mappings") / "goldstein_mappings.csv", index_col=None)
        a_map_df = goldstein_map_df[["event_code", "action_code", "action_name_short", "action_goldstein","action_goldstein_rank"]].drop_duplicates(subset="event_code")
        e_map_df = goldstein_map_df[["event_code", "event_name", "event_goldstein", "event_goldstein_rank", "event_goldstein_rank_all"]].drop_duplicates(subset="event_code")

        ## action
        df = pd.merge(df, a_map_df, how='left', left_on="E_code", right_on="event_code")
        df = df.drop(columns=["event_code"])
        df = df.rename(columns={"action_goldstein_rank": "G_A_rank", "action_goldstein": "G_A", "action_name_short": "A_name"})

        ## event
        df = pd.merge(df, e_map_df, how='left', left_on="E_code", right_on="event_code")
        df = df.drop(columns=["event_code"])
        df = df.rename(columns={"event_goldstein_rank": "G_E_rank", "event_goldstein_rank_all": "G_E_rank_all", "event_goldstein": "G_E", "event_name": "E_name"})
        return df

    def prepare_source_target(self, df, V1=[], V2=[]):
        if len(V1) > 0:
            df = df[df['V1_name'].isin(V1)]
        if len(V2) > 0:
            df = df[df['V2_name'].isin(V2)]
        return df

    def prepare_date(self, df, start_end=["2000-01-01", "2022-01-01"]):
        df["date"] = pd.to_datetime(df["date"], format='%Y-%m-%d', errors='coerce')
        df = df.loc[(df['date'] >= start_end[0]) & (df['date'] < start_end[1])]
        return df


if __name__ == '__main__':
    icews = ICEWS(file_list=[2017], V1=[], V2=[], start_end=["2000-01-01", "2022-01-01"])
    print(icews.df[icews.df.date == "2017-01-01"].count())
    print(icews.df.columns, icews.df)