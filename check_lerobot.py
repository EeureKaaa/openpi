import pandas as pd


data_path_gr00t = "/home/wangxianhao/data/project/reasoning/openpi/EeureKaaaa/gr00t_dataset/data/chunk-000/episode_000000.parquet"
data_path_pi0 = "/home/wangxianhao/data/project/reasoning/openpi/EeureKaaaa/tabletop_dataset/data/chunk-000/episode_000000.parquet"
df_gr00t = pd.read_parquet(data_path_gr00t)
df_pi0 = pd.read_parquet(data_path_pi0)
print(df_gr00t)
print(df_pi0)
