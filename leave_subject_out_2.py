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

# Learn the model parameters (W, \tau) on training videos
# On the test video, leave one subject out, 
# learn the X for the test videos using all-but-one subject
# and find the reconstruction error for held-out subject.


train = [load_ea_data(vid=v)[0] for v in [0, 1]]
test = [load_ea_data(vid=v)[0] for v in [2]]
m, q, _ = train[0].shape

# p_range = np.arange(5, 200, 5)
p_range = [160]

error = []
lscales = []

loss = []
for p in tqdm(p_range, desc='total', ncols=100):
    # train the model
    model = SharedGpfa(m, q, p, reg=0., fa_init=True)
    model.fit(
        train_data=train, 
        n_iters=100, 
        learning_rate=0.1, 
        tensorboard=False,
        desc=f'Training: p = {p}',
        ncols=100
    )

    # test the model
    err = []
    # for test_sub in tqdm(range(m), desc=f'p = {p}', ncols=100):
    for test_sub in tqdm(range(80), desc=f'p = {p}', ncols=100):
        train_subs = subs = np.arange(m) != test_sub
        x_test = model.add_video(
            test,
            subs=train_subs,
            fa_init=True,
            n_iters=100,
            learning_rate=0.1)[0][0]
        y_hat = (model.vars['w'][test_sub] @ x_test).numpy()
        y = test[0][test_sub]
        _err = np.sum((y - y_hat) ** 2)
        err.append(_err)
 
    loss.append(np.mean(err))
    # np.save('loss', loss)

plt.plot(p_range, loss)
plt.show()