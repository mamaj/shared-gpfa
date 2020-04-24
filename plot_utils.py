import matplotlib.pyplot as plt
import numpy as np

def set_xlabels_atlas(ax, label_ticks, label_names, axis_len=400, rotation=45):
    minor_xticks = []
    for i in range(len(label_ticks)):
        if i != len(label_ticks)-1: center = np.mean(label_ticks[i:i+2])
        else: center = np.mean([label_ticks[i], axis_len-1])
        minor_xticks.append(center)
    ax.set_xticks(np.append(label_ticks, axis_len-1), minor=False)
    ax.set_xticks(minor_xticks , minor=True)
    ax.set_xticklabels('', minor=False)
    ax.set_xticklabels(label_names, rotation=45, ha='right', va='top', minor=True, rotation_mode='anchor') 
    ax.tick_params(axis='x', which='minor', direction='inout', top=False, length=10)
    ax.tick_params(axis='x', which='major', direction='inout', top=False, length=2, color='w', width=5)
    ax.spines["bottom"].set_zorder(0)