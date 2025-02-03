model_args = {'depth': 12,
              'embed_dim': 768,
              'img_size': 224,
              'in_chans': 6,
              'num_frames': 1,
              'num_heads': 12,
              'patch_size': 16,
              'tubelet_size': 1}

data_mean = [775.2290211032589, 1080.992780391705, 1228.5855250417867, 2497.2022620507532, 2204.2139147975554,
             1610.8324823273745]

data_std = [1281.526139861424, 1270.0297974547493, 1399.4802505642526, 1368.3446143747644, 1291.6764008585435,
            1154.505683480695]

data_mean_phi2_small = [0.1072133 , 0.10215581, 0.09828673, 0.12190109, 0.19228693, 0.22711556]

data_mean_phi2_small = [0.04663816, 0.04991332, 0.07230965, 0.07187664, 0.08743845, 0.10470329]

data_mean_phi2 = [value * 10000 for value in data_mean_phi2_small]
data_std_phi2 = [value * 10000 for value in data_mean_phi2_small]

