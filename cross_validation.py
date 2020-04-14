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


# Leave Subjects Out

# data, subdf, score = load_ea_data(vid=0)
# m, q, t = data.shape

# n_splits = 5
# p_range = np.arange(1, 200, 5)

# kf = KFold(n_splits=n_splits, shuffle=True, random_state=1)

# error = []
# for p in tqdm(p_range):
#     print(f'started to evaluate p = {p}')
#     error.append([])
#     for train, test in tqdm(kf.split(data), total=n_splits):
#         model = SharedGpfa(len(train), q, p, t)
#         model.fit(
#             train_data=data[train], 
#             n_iters=1200, 
#             learning_rate=0.075, 
#             tensorboard=False
#         )
#         _, rec_error = model.add_subject(data[test])
#         error[-1].append(rec_error)
#     np.save('error', np.array(error))

# error = np.array(error)
# np.save('error', error)
# plt.plot(p_range, error.mean(-1), marker='.')
# plt.show()




# Other Videos for validation

train, _, _ = load_ea_data(vid=0)
test, _, _ = load_ea_data(vid=1)

m, q, t_train = train.shape
_, _, t_test = test.shape

p_range = np.arange(2, 200, 2)
# p_range = np.array([5, 10, 20])

error = []
lscales = []

ncols = 4
nrows = math.ceil(p_range.size / ncols)
fig, axs = plt.subplots(nrows, ncols, figsize=(ncols*5, nrows*5))
fig_bar, ax_bar = plt.subplots()
plt.ion()
plt.show()

for i, p in enumerate(tqdm(p_range, desc='total', ncols=100)):
    model = SharedGpfa(m, q, p, t_train)
    model.fit(
        train_data=train, 
        n_iters=2000, 
        learning_rate=0.05, 
        tensorboard=False,
        desc=f'p = {p}',
        ncols=100
    )

    length_scale = tf.convert_to_tensor(model.vars['length_scale']).numpy()
    lscales.append(length_scale)
    axs.ravel()[i].bar(np.arange(model.p), length_scale)

    x = model.add_video(test)
    recon_error = tf.reduce_sum((test - model.vars['w'] @ x) ** 2)
    error.append(recon_error)
    np.save('error', error)
    np.save('lscale', lscales)
    ax_bar.plot(p_range[:i+1], error, marker='.')
    ax_bar.set_title(f'p = {p}')

    plt.draw()
    plt.pause(0.05)

plt.show(block=True)