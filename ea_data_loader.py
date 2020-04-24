import sys
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd
from scipy import stats


def load_ea_data(vid, platform='reza_win', normalize=True, filter_subs=True, circle_quantile=0.01, fdmean=0.5, fdperc=50):
    data_path = get_data_path(platform)
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

def get_atlas_ticks(platform='reza_win'):
    data_path = get_data_path(platform)
    yeo = pd.read_csv(data_path / 'yeo17labels.csv')
    label_ticks = np.where(yeo.region != yeo.region.shift())[0]
    label_names = (yeo.hemi[label_ticks] + ' ' + yeo.region[label_ticks]).to_numpy()
    return yeo, label_ticks, label_names
    
def get_data_path(platform='reza_win'):
    if platform == 'reza_wsl':
        data_path = Path('/mnt/c/Users/mamar/OneDrive - University of Toronto/Thesis/Brain Network/SPINS/SPINS_ciftify/schaefer_ea')
    elif platform == 'reza_win':
        data_path = PureWindowsPath(r"C:\Users\mamar\OneDrive - University of Toronto\Thesis\Brain Network\SPINS\SPINS_ciftify\schaefer_ea")
    elif platform == 'camh':
        data_path = Path('/projects/rebrahimi/GSP/SPINS_ciftify/schaefer_ea/')
    else:
        data_path = None
    return data_path

def get_fsavg_map(platform='camh'):
    if platform == 'camh':
        sch2fsavg = ['/projects/rebrahimi/GSP/SPINS_fMRIprep/Parcellations/FreeSurfer5.3/fsaverage5/label/lh.Schaefer2018_400Parcels_17Networks_order.annot',
                     '/projects/rebrahimi/GSP/SPINS_fMRIprep/Parcellations/FreeSurfer5.3/fsaverage5/label/rh.Schaefer2018_400Parcels_17Networks_order.annot'
                    ]
    else:
        sch2fsavg = None
    return sch2fsavg