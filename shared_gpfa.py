import datetime
import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import tqdm
from sklearn.decomposition import FactorAnalysis

tfd = tfp.distributions
tfb = tfp.bijectors
dtype = np.float32


class SharedGpfa:

    def __init__(self, m, q, p, latent_noise_var=1e-4, dtype=np.float32):
        self.m = m
        self.q = q
        self.p = p
        self.t = None
        self.latent_noise_var = latent_noise_var
        self.amplitude = math.sqrt(1 - latent_noise_var)
        self.dtype = dtype
        self.vars = dict()
        self.joint = []

    def init_vars(self):
        self.check_model_fit()
        def init_log(size):
            return np.random.lognormal(size=size).astype(self.dtype)
        def init_norm(size): 
            return np.random.normal(size=size).astype(self.dtype)
        def init_uniform(size): 
            return np.random.uniform(size=size).astype(self.dtype)
        constrain_positive = tfb.Shift(np.finfo(np.float32).tiny)(tfb.Exp())

        self.vars['w'] = tf.Variable(
            init_uniform((self.m, self.q, self.p)),
            name='w'
        )
        self.vars['length_scale'] = tfp.util.TransformedVariable(
            init_log((self.p)),
            constrain_positive,
            name='length_scale'
        )
        self.vars['subject_noise_scale'] = tfp.util.TransformedVariable(
            init_log(self.m), 
            constrain_positive, 
            name='subject_noise_scale'
        )
        self.vars['roi_noise_scale'] = tfp.util.TransformedVariable(
            init_log(self.q), 
            constrain_positive, 
            name='roi_noise_scale'
        )
        self.vars['x'] = []
        for i, t in enumerate(self.t):
            x = tf.Variable(init_uniform((self.p, t)), name='x{i}')
            self.vars['x'].append(x)
        self.vars['amplitude'] = tfp.util.TransformedVariable(
            init_log(self.p), 
            constrain_positive, 
            name='amplitude'
        )

    def init_joint(self):
        self.check_model_fit()
        for t in self.t:
            self.joint.append(self.create_joint(t))

    # needs checking
    def create_joint(self, t, subs=Ellipsis):
        ind_points = np.arange(t).astype(self.dtype)
        joint = tfd.JointDistributionNamed(dict(
            x = tfd.Independent(tfd.GaussianProcess(
                kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(
                    amplitude = self.amplitude,
                    # amplitude = self.vars['amplitude'], 
                    length_scale = self.vars['length_scale'],
                    feature_ndims = 1
                ),
                index_points = ind_points[:, None],
                jitter = 1e-4,
                observation_noise_variance = self.latent_noise_var,
                validate_args = True
            ), 1),
            obs = lambda x:
                tfd.Independent(tfd.Normal(
                    loc = tf.matmul(self.vars['w'][subs], tf.expand_dims(x, -3)),
                    scale = tf.expand_dims(tf.expand_dims(
                        self.vars['subject_noise_scale'][subs], -1) * self.vars['roi_noise_scale'], -1)
                ), 3)
        ))
        return joint


    def create_joint_reg(self, t, reg, subs=Ellipsis):
        ind_points = np.arange(t).astype(self.dtype)
        joint = tfd.JointDistributionNamed(dict(
            
            x = tfd.Independent(tfd.GaussianProcess(
                kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(
                    amplitude = self.vars['amplitude'],
                    length_scale = self.vars['length_scale'],
                    feature_ndims = 1
                ),
                index_points = ind_points[:, None],
                jitter = 1e-4,
                observation_noise_variance = self.latent_noise_var,
                validate_args = True
            ), 1),

            w = tfd.Independent(tfd.Laplace(
                loc = tf.zeros(self.vars['w'].shape),
                scale = 1 / reg
            )),

            obs = lambda x, w:
                tfd.Independent(tfd.Normal(
                    loc = tf.matmul(w[subs], tf.expand_dims(x, -3)),
                    scale = tf.expand_dims(tf.expand_dims(
                        self.vars['subject_noise_scale'][subs], -1) * self.vars['roi_noise_scale'], -1)
                ), 3)
        ))
        return joint


    def log_prob(self, data, joint=None, x=None):
        self.check_model_fit()
        if joint is None:
            joint = self.joint
        if x is None:
            x = self.vars['x']
        data = self.add_batch(data)

        log_prob = []
        for _joint, _x, _data in zip(joint, x, data):
            log_prob.append(
                _joint.log_prob(dict(
                    x=_x,
                    obs=_data
                )))
        return tf.reduce_sum(log_prob)

    def l1_loss(self):
        return tf.reduce_mean(tf.abs(self.vars['w']))

    def fit(self, train_data, n_iters, learning_rate=0.01, reg=0., tensorboard=False, fa_init=True, **kwargs):

        train_data = self.add_batch(train_data)
        self.t = [s.shape[-1] for s in train_data]
        self.init_vars()
        if fa_init:
            self.assign_fa(train_data)
        self.init_joint()

        opt = tf.optimizers.Adam(learning_rate=learning_rate)
        if tensorboard:
            self.setup_tfboard()
        trainable_vars = self.get_trainable_vars()

        @tf.function
        def loss():
            return -self.log_prob(train_data) / (sum(self.t) * (self.p + self.q * self.m))
            + reg * self.l1_loss()

        @tf.function
        def train_step():
            opt.minimize(loss, trainable_vars)

        l = []
        for epoch in tqdm(range(int(n_iters)), **kwargs):
            train_step()
            l.append(loss())
            if tensorboard:
                self.update_tfsummary(loss(), epoch)
        return l

    def add_video(self, obs, n_iters=1e3, learning_rate=0.04, subs=Ellipsis, name='new_x', tensorboard=False, fa_init=True, **kwargs):

        if subs is None:
            subs = range(self.m)

        obs = self.add_batch(obs)
        obs_subs = []
        for s in obs:
            obs_subs.append(s[subs])
        t = [s.shape[-1] for s in obs_subs]
        new_joint = [self.create_joint(_t, subs=subs) for _t in t]

        if fa_init:
            xlist_pinv = self.least_square(obs)
        else:
            def init(size): return np.random.normal(size=size).astype(self.dtype)

        xlist = []
        for i, _t in enumerate(t):
            if fa_init:
                x_init = xlist_pinv[i]
            else:
                x_init = init((self.p, _t))
            x = tf.Variable(x_init, name=name+f'_{i}', dtype=dtype)
            xlist.append(x)

        if tensorboard:
            self.setup_tfboard()
        opt = tf.optimizers.Adam(learning_rate=learning_rate)

        @tf.function
        def loss():
            return -self.log_prob(obs_subs, new_joint, xlist) / (sum(t) * (self.p + self.q * self.m))
            + self.l1_loss()

        @tf.function
        def recon_loss():
            return tf.reduce_sum([tf.reduce_sum((_obs - self.vars['w'][subs] @ _x) ** 2) for _obs, _x in zip(obs_subs, xlist)])

        @tf.function
        def train_step():
            opt.minimize(loss, xlist)

        # l = []
        for epoch in range(int(n_iters)):
            train_step()
            if tensorboard:
                self.update_tfsummary(loss(), epoch)
                with self.summary_writer.as_default():
                    for i, x in enumerate(xlist):
                        tf.summary.histogram(f'xlist{i}_0', x[0], step=epoch)
                        tf.summary.histogram(f'xlist{i}_1', x[1], step=epoch)
            # l.append(loss())
        return xlist, recon_loss()

    def add_subject(self, obs, experiment, n_iters=1e3, learning_rate=0.04, name='w_new'):
        if obs.ndim == 2:
            obs = np.expand_dims(obs, 0)

        def init_norm(size): return np.random.normal(
            size=size).astype(self.dtype)
        w = tf.Variable(init_norm((obs.shape[0], self.q, self.p)), name=name)
        opt = tf.optimizers.Adam(learning_rate=learning_rate)

        @tf.function
        def loss():
            return tf.reduce_sum((obs - w @ self.vars['x'][experiment]) ** 2)

        @tf.function
        def train_step():
            opt.minimize(loss, (w))

        for i in range(int(n_iters)):
            train_step()
        return w, loss()

    def factor_analysis(self, data, **kwargs):

        fa = FactorAnalysis(n_components=self.p, random_state=0, **kwargs)

        y = np.concatenate(data, -1)
        y = np.concatenate(y, 0)
        y = y.transpose()

        x = fa.fit_transform(y)

        split_ind = np.cumsum([d.shape[-1] for d in data])[:-1]
        x = x.transpose()
        x = np.array_split(x, split_ind, axis=-1)
        w = fa.components_.transpose()
        w = np.split(w, data[0].shape[0])
        w = np.stack(w, axis=0)

        return x, w

    def assign_fa(self, data):
        if sum(d.shape[-1] for d in data) < self.p:
            warnings.warn("p > t, cannot use FA for initialization.")
            return

        x_fa, w_fa = self.factor_analysis(data)
        for _x, _x_fa in zip(self.vars['x'], x_fa):
            _x.assign(_x_fa)
        self.vars['w'].assign(w_fa)

    def least_square(self, data):
        w = self.vars['w'].numpy()
        w = np.concatenate(w, axis=0)
        w_pinv = np.linalg.pinv(w)
        xlist = []
        for y in data:
            x_hat = w_pinv @ np.concatenate(y, axis=0)
            xlist.append(x_hat)
        return xlist

    def setup_tfboard(self):
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = './logs/' + current_time
        self.summary_writer = tf.summary.create_file_writer(train_log_dir)

    def update_tfsummary(self, loss, epoch):
        with self.summary_writer.as_default():
            tf.summary.scalar('loss', loss, step=epoch)
            tf.summary.histogram('w0', self.vars['w'][0], step=epoch)
            tf.summary.histogram('w1', self.vars['w'][1], step=epoch)
            tf.summary.histogram('lenth_scale', self.vars['length_scale'], step=epoch)
            for i, x in enumerate(self.vars['x']):
                tf.summary.histogram(f'x_sample{i}_0', x[0], step=epoch)
                tf.summary.histogram(f'x_sample{i}_1', x[1], step=epoch)
            for i, lenscale in enumerate(self.vars['length_scale']):
                tf.summary.scalar(f'lenth_scale_{i}',lenscale, step=epoch)
            # tf.summary.scalar('lenscale0', self.vars['length_scale'][0], step=epoch)
            # tf.summary.scalar('lenscale1', self.vars['length_scale'][1], step=epoch)

    def get_trainable_vars(self, only_train_x=False):
        trainable_vars = []
        trainable_vars += self.vars['x']
        if not only_train_x:
            trainable_vars.append(self.vars['w'])
            transformed_vars = (
                'length_scale', 
                'subject_noise_scale', 
                'roi_noise_scale'
                # 'amplitude'
            )
            for v in transformed_vars:
                trainable_vars.append(self.vars[v].trainable_variables[0])
        return trainable_vars

    def check_model_fit(self):
        assert self.t is not None, "model.fit() is not passed yet"

    def add_batch(self, data):
        if type(data) is not list and data.ndim == 3:
            return [data]
        else:
            return data
