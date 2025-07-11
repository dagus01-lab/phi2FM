import json
import os
import pandas as pd

df_simulated = pd.read_csv('/home/ccollado/1_simulate_data/Major-TOM/df_simulation.csv')
df = df_simulated['unique_identifier']

np_files = os.listdir('/home/ccollado/phileo_phisat2/MajorTOM/np_patches_128_cropped')
train_files = [f.replace('_train_s2.npy', '') for f in np_files if f.endswith('train_s2.npy')]
test_files = [f.replace('_test_s2.npy', '') for f in np_files if f.endswith('test_s2.npy')]
len(train_files), len(test_files)


df_train = df[df.isin(train_files)]
df_test = df[df.isin(test_files)]

df_train = df_train + '_train_s2.npy'
df_test = df_test + '_test_s2.npy'

data = {
    "train": df_train.tolist(),
    "test": df_test.tolist()
}

with open('/home/ccollado/1_simulate_data/Major-TOM/train_test_split.json', 'w') as f:
    json.dump(data, f)