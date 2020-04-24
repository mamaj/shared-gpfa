import sys
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd
from scipy import stats


def load_ea_data(vid, normalize=True, filter_subs=True, circle_quantile=0.01, fdmean=0.5, fdperc=50):
    if 'linux' in sys.platform:
        data_path = Path('/mnt/c/Users/mamar/OneDrive - University of Toronto/Thesis/Brain Network/SPINS/SPINS_ciftify/schaefer_ea')
    else:
        data_path = PureWindowsPath(r"C:\Users\mamar\OneDrive - University of Toronto\Thesis\Brain Network\SPINS\SPINS_ciftify\schaefer_ea")

    subdf = pd.read_csv(data_path / 'subjects_df.csv')
    vid_dur = pd.read_csv(data_path / 'vid_dur.csv').to_numpy()[0]
    
    with np.load(data_path / "schaefer_ea.npz") as data:
        task = data['task']
    
    if normalize:
        task = zscore(task)

    if filter_subs:
        subs2remove = get_filtered_subs(subdf, circle_quantile, fdmean, fdperc)
        subdf = subdf.loc[~subs2remove, :]
        task = task[~subs2remove]

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

def get_filtered_subs(subdf, circle_quantile, fdmean, fdperc):

    circle = np.arctanh(subdf.filter(regex='circle'))
    score_filter = circle.le(circle.quantile(circle_quantile)).any(axis=1)
    fdmean_filter = subdf.filter(regex='fd_mean_run').ge(fdmean).any(axis=1)
    fdperc_filter = subdf.filter(regex='fd_perc_run').ge(fdperc).any(axis=1)

    return (score_filter | fdmean_filter | fdperc_filter)
