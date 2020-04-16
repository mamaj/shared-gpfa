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

# Uses only one videos
# Learn the model parameters and X on training subjects
# Learn the W for test subjects and report reconstruction loss

data, subdf, score = load_ea_data(vid=0)
m, q, t = data.shape

n_splits = 5
p_range = np.arange(1, 200, 5)

kf = KFold(n_splits=n_splits, shuffle=True, random_state=1)

error = []
for p in tqdm(p_range):
    print(f'started to evaluate p = {p}')
    error.append([])
    for train, test in tqdm(kf.split(data), total=n_splits):
        model = SharedGpfa(len(train), q, p, t)
        model.fit(
            train_data=data[train], 
            n_iters=1200, 
            learning_rate=0.075, 
            tensorboard=False
        )
        _, rec_error = model.add_subject(data[test])
        error[-1].append(rec_error)
    np.save('error', np.array(error))

error = np.array(error)
np.save('error', error)
plt.plot(p_range, error.mean(-1), marker='.')
plt.show()
