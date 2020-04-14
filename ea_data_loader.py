from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def load_ea_data(vid, normalize=True):
    data_path = Path(r"C:\Users\mamar\OneDrive - University of Toronto\Thesis\Brain Network\SPINS\SPINS_ciftify\schaefer_ea")
    subdf = pd.read_csv(data_path / 'subjects_df.csv')
    data = np.load(data_path / "schaefer_ea.npz")
    vid_dur = pd.read_csv(data_path / 'vid_dur.csv').to_numpy()[0]
    
    task = data['task']

    if normalize:
        task = zscore(task)

    onset = subdf.filter(regex='onset').iloc[:, vid]
    data = [bold[:, int(start):int(start + vid_dur[vid])] for bold, start in zip(task, onset)]
    t = min(d.shape[-1] for d in data)
    data = np.stack([d[:, :t] for d in data])

    score = np.arctanh(subdf.filter(regex='score').iloc[:,vid])

    return data, subdf, score

def zscore(task):
    ztask = []
    for run in np.split(task, 3, axis=-1):
        ztask.append(stats.zscore(run, axis=-1))
    return np.concatenate(ztask, axis=-1)
