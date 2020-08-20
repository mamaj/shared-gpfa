import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import KFold
from shared_gpfa import SharedGpfa
from ea_data_loader import load_ea_data
import time
import pickle
import math

tf.get_logger().setLevel('INFO')





# Learn params (W, \tau, \sigma) on some videos,
# find x for test videos and report the reconstruction loss

train = [load_ea_data(vid=v)[0] for v in [0, 1, 2, 3, 4, 5]]
m, q, _ = train[0].shape
test = [load_ea_data(vid=v)[0] for v in [6, 7, 8]]

p_range = np.arange(5, 200, 5)

error = []
lscales = []

ncols = 4
nrows = math.ceil(p_range.size / ncols)
fig, axs = plt.subplots(nrows, ncols, figsize=(ncols*5, nrows*5))
fig_bar, ax_bar = plt.subplots()
plt.ion()
plt.show()

for i, p in enumerate(tqdm(p_range, desc='total', ncols=100)):
    model = SharedGpfa(m, q, p, reg=0., fa_init = True)
    model.fit(
        train_data=train, 
        n_iters=1200, 
        learning_rate=0.075, 
        tensorboard=False,
        desc=f'p = {p}',
        ncols=100
    )

    length_scale = tf.convert_to_tensor(model.vars['length_scale']).numpy()
    lscales.append(length_scale)
    axs.ravel()[i].bar(np.arange(model.p), length_scale)

    x, recon_error = model.add_video(test)
    # recon_error = tf.reduce_sum((test - model.vars['w'] @ x) ** 2)
    error.append(recon_error)
    np.save('error', error)
    np.save('lscale', lscales)
    ax_bar.plot(p_range[:i+1], error, marker='.')
    ax_bar.set_title(f'p = {p}')

    plt.draw()
    plt.pause(0.05)

plt.show(block=True)