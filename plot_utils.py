import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from ea_data_loader import get_atlas_ticks
from nilearn import plotting, datasets
import nibabel as nib


def set_xlabels_atlas(ax, label_ticks, label_names, axis_len=400, rotation=45):
    minor_xticks = []
    for i in range(len(label_ticks)):
        if i != len(label_ticks)-1: center = np.mean(label_ticks[i:i+2])
        else: center = np.mean([label_ticks[i], axis_len-1])
        minor_xticks.append(center)
    ax.set_xticks(label_ticks[1:], minor=False)
    ax.set_xticks(minor_xticks , minor=True)
    ax.set_xticklabels('', minor=False)
    ax.set_xticklabels(label_names, rotation=rotation, ha='right', va='top', minor=True, rotation_mode='anchor') 
    ax.tick_params(axis='x', which='minor', direction='inout', top=False, length=10)
    ax.tick_params(axis='x', which='major', direction='inout', top=False, length=2, color='w', width=5)
    ax.spines["bottom"].set_zorder(0)
    
    
# functions for Brain plots

def plot_all_brain_views(stat, hemi, platform='camh', fsaverage='fsaverage5', fontsize=14, **kwargs):
    view = ['lateral', 'medial', 'dorsal', 'ventral', 'anterior', 'posterior']
    fig, axs = plt.subplots(2, 3, 
                            figsize=(20,10),
                            facecolor=(1,1,1,0), 
                            subplot_kw={'projection': '3d', 'facecolor':(1,1,1,0)})
    plt.subplots_adjust(wspace=-.3, hspace=-.3)
    for i, v in enumerate(view):
        ax = axs.ravel()[i]
        surf_plot(stat, hemi=hemi, view=v, platform=platform, axes=ax, colorbar=(i==0), fsaverage=fsaverage, **kwargs)
        t = ax.set_title(v,  fontdict={'fontsize':fontsize})
        if i > 2:
            t.set_y(0.08)
    cax = fig.findobj(plt.Axes)[-1]
    cax.set_position([0.1, 0.38, 0.5, 0.3])
    cax.set_frame_on(True)
    plt.setp(cax.spines.values(), color='gray', lw=0.5)
    return fig, axs

def plot_brain_hemis(stat, platform='camh', fsaverage='fsaverage5', figsize=(15, 10), **kwargs):
    view = ['lateral', 'medial']
    fig, axs = plt.subplots(2, 2,  figsize=figsize, facecolor=(1,1,1,0),
                            subplot_kw={'projection': '3d', 'facecolor':(1,1,1,0)})
    plt.subplots_adjust(wspace=-.25, hspace=-.25)
    surf_plot(stat, hemi='left',  view=view[0], axes=axs[0, 0], platform=platform, colorbar=True,  fsaverage=fsaverage, **kwargs)
    surf_plot(stat, hemi='left',  view=view[1], axes=axs[1, 0], platform=platform, colorbar=False, fsaverage=fsaverage, **kwargs)
    surf_plot(stat, hemi='right', view=view[0], axes=axs[0, 1], platform=platform, colorbar=False, fsaverage=fsaverage, **kwargs)
    surf_plot(stat, hemi='right', view=view[1], axes=axs[1, 1], platform=platform, colorbar=False, fsaverage=fsaverage, **kwargs)
    cax = fig.findobj(plt.Axes)[-1]
    cax.set_position([0.1, 0.38, 0.5, 0.3])
    cax.set_frame_on(True)
    plt.setp(cax.spines.values(), color='gray', lw=0.5)
    return fig, axs

def surf_plot(stat, hemi='left', mesh='infl', platform='camh', fsaverage='fsaverage5', **kwargs):
    stat_fsavg, fsaverage = sch2fsavg(stat, mesh=fsaverage)
    fig = plotting.plot_surf(
        fsaverage[mesh + '_' + hemi],
        stat_fsavg[0 if hemi=='left' else 1], 
        hemi = hemi,
        bg_map = fsaverage['sulc_' + hemi],
        bg_on_data=True,
        **kwargs)
    return fig

def surf_plot_roi(stat, hemi='left', mesh='infl', platform='camh', fsaverage='fsaverage5', **kwargs):
    stat_fsavg, fsaverage = sch2fsavg(stat, mesh=fsaverage)
    fig = plotting.plot_surf_roi(
        fsaverage[mesh + '_' + hemi],
        stat_fsavg[0 if hemi=='left' else 1], 
        hemi = hemi,
        bg_map = fsaverage['sulc_' + hemi],
        bg_on_data=True,
        **kwargs)
    return fig



def get_sch2fsavg_map(platform='camh', parc=400, net='17', mesh='fsaverage5'):
    if platform == 'camh':
        map_path = Path(f'/projects/rebrahimi/GSP/SPINS_fMRIprep/Parcellations/FreeSurfer5.3/{mesh}/label/')
    else:
        return None
    fsavg_map = []
    for hemi in ('l', 'r'):
        sch2fsavg = map_path / f'{hemi[0]}h.Schaefer2018_{parc}Parcels_{net}Networks_order.annot' 
        schaefer_hemi = nib.freesurfer.io.read_annot(sch2fsavg)
        fsavg_map.append(dict())
        fsavg_map[-1]['map'] = schaefer_hemi[0]
        fsavg_map[-1]['labels'] = [l.decode('UTF-8') for l in schaefer_hemi[2]]
    # check my atlas and fsaverage mapping orders are the same
    map_label =  fsavg_map[0]['labels'][1:] + fsavg_map[1]['labels'][1:]
    yeo, _, _ = get_atlas_ticks(platform)
    assert (yeo.labelname == map_label).all()
    return fsavg_map


def sch2fsavg(stat_sch, parc=400, net='17', mesh='fsaverage5', platform='camh'):
    fsavg_map = get_sch2fsavg_map(platform='camh', parc=400, net='17', mesh=mesh)
    stat_sch = np.array_split(stat_sch, 2)
    stat_fsavg = []
    for hemi in range(2):
        mapping = fsavg_map[hemi]['map']
        stat_sch_hemi = stat_sch[hemi]
        stat_sch_hemi = np.insert(stat_sch_hemi, 0, np.nan)
        stat_fsavg_hemi = stat_sch_hemi[mapping]
        stat_fsavg.append(stat_fsavg_hemi)
    fsaverage = datasets.fetch_surf_fsaverage(mesh=mesh)
    return stat_fsavg, fsaverage


# Class for colormap manipulation

class CmapTool():
    def __init__(self, cmap, vmin=0, vmax=1, cmlen=256):
        self.cmap = mpl.cm.get_cmap(cmap, cmlen)
        self.vmin = vmin
        self.vmax = vmax
        self.cmlen = cmlen
    
    def to_cmapidx(self, val):
        norm = mpl.colors.Normalize(vmin=self.vmin, vmax=self.vmax)
        return int(norm(val) * self.cmlen)
    
    def clip_outside(self, cmin, cmax, tr=None):
        if tr is None:
            tr = np.array((1, 1, 1, 1))
        colors = self.cmap(np.linspace(0, 1, self.cmlen))
        colors[self.to_cmapidx(self.vmin):self.to_cmapidx(cmin), :] = tr
        colors[self.to_cmapidx(cmax): self.to_cmapidx(self.vmax), :] = tr
        self.cmap = mpl.colors.ListedColormap(colors)
        return self
    
    def clip_inside(self, cmin, cmax, tr=None):
        if tr is None:
            tr = np.array((1, 1, 1, 1))
        colors = self.cmap(np.linspace(0, 1, self.cmlen))
        colors[self.to_cmapidx(cmin): self.to_cmapidx(cmax), :] = tr
        self.cmap = mpl.colors.ListedColormap(colors)
        return self
         
    def symmetrize(self):
        colors = self.cmap(np.linspace(0, 1, self.cmlen))
        colors = np.vstack((colors[::-1], colors))
        self.cmap = mpl.colors.ListedColormap(colors)
        return self
    
    def change_center(self, center):
        colors = self.cmap(np.linspace(0, 1, self.cmlen))
        colors = np.array_split(colors, 2)
        low_cmap = mpl.colors.ListedColormap(colors[0])
        high_cmap = mpl.colors.ListedColormap(colors[1])
        
        low_colors = low_cmap(np.linspace(0, 1, self.to_cmapidx(center)))
        high_colors = high_cmap(np.linspace(0, 1, self.cmlen - self.to_cmapidx(center)))
        new_colors = np.concatenate((low_colors, high_colors), axis=0)
        
        self.cmap = mpl.colors.ListedColormap(new_colors)
        return self
    
    def reverse(self):
        self.cmap = self.cmap.reversed()
        return self

    def set_nan(self, color='white', alpha=1):
        self.cmap.set_bad(color=color, alpha=alpha)
        return self

    
# Serializing figures

def save_fig(fig, fname):
    if fname.find('.') == -1:
        fname += '.pkl'
    with open(fname, 'wb') as fid:
        pickle.dump(fig, fid)

        
def load_fig(fname):
    if fname.find('.') == -1:
        fname += '.pkl'
    with open(fname, 'rb') as fid:
        fig = pickle.load(fid)
    return fig


# Spines colors
def spine(fig, color='lightgray', lw=1.5):
    for ax in fig.get_axes():
        for s in ax.spines.values():
            s.set_color(color)
            s.set_lw(lw)

            
# figure font size
def font_size(fig, fs):
    for ax in fig.get_axes():
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels() + ax.get_xticklabels(minor=True)):
            item.set_fontsize(fs)