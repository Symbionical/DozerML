import mne
import datashaper
import pandas as pd

def extract_frame(_path):
    raw = mne.io.read_raw_cnt(_path)
    _df = raw.to_data_frame()
    _df = datashaper.downsample_and_trim(_df)
    _df = datashaper.get_channels(_df,'T3','O2')
    return _df

def create_normal_dataset():
    df_norm_1 = extract_frame("TrainingData/1/Normal state.cnt")
    df_norm_2 = extract_frame("TrainingData/2/Normal state.cnt")
    df_norm_3 = extract_frame("TrainingData/3/Normal state.cnt")
    df_norm_4 = extract_frame("TrainingData/4/Normal state.cnt")
    df_norm_5 = extract_frame("TrainingData/5/Normal state.cnt")
    df_norm_6 = extract_frame("TrainingData/6/Normal state.cnt")
    df_norm_7 = extract_frame("TrainingData/7/Normal state.cnt")
    df_norm_8 = extract_frame("TrainingData/8/Normal state.cnt")
    df_norm_9 = extract_frame("TrainingData/9/Normal state.cnt")
    df_norm_10 = extract_frame("TrainingData/10/Normal state.cnt")
    df_norm_11 = extract_frame("TrainingData/11/Normal state.cnt")
    df_norm_12 = extract_frame("TrainingData/12/Normal state.cnt")

    normal_frames = [

        df_norm_1,
        df_norm_2,
        df_norm_3,
        df_norm_4,
        df_norm_5,
        df_norm_6,
        df_norm_7,
        df_norm_8,
        df_norm_9,
        df_norm_10,
        df_norm_11,
        df_norm_12

    ]

    df_normal = pd.concat(normal_frames)
    df_normal.to_pickle("data_normal.pkl")

def create_normal_FuzEn_dataset(_normal_df):
    _df_FuzEn = datashaper.df_raw_to_entropy(_normal_df, 'T3', 'O2')
    _df_FuzEn.to_pickle("FuzEn_data_normal.pkl")


create_normal_dataset()

normal_data = pd.read_pickle("data_normal.pkl")

create_normal_FuzEn_dataset(normal_data)

FuzEn_data_normal = pd.read_pickle("data_normal_FuzEn.pkl")

print(FuzEn_data_normal)
