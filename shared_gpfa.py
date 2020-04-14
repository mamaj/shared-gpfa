import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import tqdm

tfd = tfp.distributions
tfb = tfp.bijectors
dtype = np.float32

class SharedGpfa:

    def __init__(self, m, q, p, t, latent_noise_var=1e-4, dtype=np.float32, reg=0.):
        self.m = m
        self.q = q
        self.p = p
        self.t = t
        self.reg = reg
        self.latent_noise_var = latent_noise_var
        self.amplitude = 1 - latent_noise_var
        self.dtype = dtype
        self.vars = dict()
        self.init_vars()
        # self.init_joint()


    def init_vars(self):
        init_log = lambda size: np.random.lognormal(size=size).astype(self.dtype)
        init_norm = lambda size: np.random.normal(size=size).astype(self.dtype)
        init_uniform = lambda size: np.random.uniform(size=size).astype(self.dtype)

        constrain_positive = tfb.Shift(np.finfo(np.float32).tiny)(tfb.Exp())

        self.vars['w'] = tf.Variable(init_uniform((self.m, self.q, self.p)), name='w')
        self.vars['length_scale'] = tfp.util.TransformedVariable(init_log((self.p)), constrain_positive, name='length_scale')
        self.vars['subject_noise_scale'] = tfp.util.TransformedVariable(init_log(self.m), constrain_positive, name='subject_noise_scale')
        self.vars['roi_noise_scale'] = tfp.util.TransformedVariable(init_log(self.q), constrain_positive, name='roi_noise_scale')
        
        # self.vars['x'] = tf.Variable(init_uniform((self.p, self.t)), name='x')
        

    def init_joint(self):
        self.joint = self.create_joint(self.t)


    def create_joint(self, t):
        ind_points = np.arange(t).astype(self.dtype)
        joint = tfd.JointDistributionNamed(dict(
            x = tfd.Independent(tfd.GaussianProcess(
                kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(self.amplitude, self.vars['length_scale'], feature_ndims=1),
                index_points = ind_points[:, None],
                jitter = 1e-4,
                observation_noise_variance = self.latent_noise_var,
                validate_args = True
            ), 1),
            obs = lambda x:
                tfd.Independent(tfd.Normal(
                    loc = tf.matmul(self.vars['w'], tf.expand_dims(x, -3)),
                    scale = tf.expand_dims(tf.expand_dims(self.vars['subject_noise_scale'], -1) * self.vars['roi_noise_scale'] , -1)
                ), 3)
        ))
        return joint


    def log_prob(self, data):
        log_prob = self.joint.log_prob(dict(
            x = self.vars['x'],
            obs = data
        ))
        return tf.reduce_mean(log_prob)


    def l1_loss(self):
        return self.reg * tf.reduce_mean(tf.abs(self.vars['w']))


    def fit(self, train_data, n_iters, learning_rate=0.01, tensorboard=True, **kwargs):
        opt = tf.optimizers.Adam(learning_rate=learning_rate)

        @tf.function
        def loss():
            return -self.log_prob(train_data) / (self.m * self.t * self.q * self.p) + self.l1_loss()

        @tf.function
        def train_step():
            opt.minimize(loss, trainable_vars)

        trainable_vars = [self.vars[t].trainable_variables[0] for t in ('length_scale', 'subject_noise_scale', 'roi_noise_scale')]
        trainable_vars += [self.vars[t] for t in ('x', 'w')]

        if tensorboard:
            summary_writer = self.setup_tfboard()
        l = []
        for epoch in tqdm(range(int(n_iters)), **kwargs):
            train_step()
            l.append(loss())
            if tensorboard:
                with summary_writer.as_default():
                    tf.summary.scalar('loss', l[-1], step=epoch)
                    tf.summary.histogram('w0', self.vars['w'][0], step=epoch)
                    tf.summary.histogram('w1', self.vars['w'][1], step=epoch)
                    tf.summary.histogram('x0', self.vars['x'][0], step=epoch)
                    tf.summary.histogram('x1', self.vars['x'][1], step=epoch)
                    tf.summary.scalar('lenscale0', self.vars['length_scale'][0], step=epoch)
                    tf.summary.scalar('lenscale1', self.vars['length_scale'][1], step=epoch)
        return l


    def add_subject(self, obs, n_iters=1e3, learning_rate=0.04, name='w_new'):
        if obs.ndim == 2:
            obs = np.expand_dims(obs, 0)
        init_norm = lambda size: np.random.normal(size=size).astype(self.dtype)
        w = tf.Variable(init_norm((obs.shape[0], self.q, self.p)), name=name)
        opt = tf.optimizers.Adam(learning_rate=learning_rate)

        @tf.function
        def loss():
            return tf.reduce_sum((obs - w @ self.vars['x']) ** 2)
            
        @tf.function
        def train_step():
            opt.minimize(loss, (w))

        for i in range(int(n_iters)):
            train_step()
        return w, loss()


    def add_video(self, obs, n_iters=1e3, learning_rate=0.04, name='new_x'):

        assert obs.shape[-2] == self.q, "number of observation timeseries does not match."
        assert obs.shape[-3] == self.m, "number of observation subjects does not match."
        
        t = obs.shape[-1]

        init_uniform = lambda size: np.random.uniform(size=size).astype(self.dtype)
        x = tf.Variable(init_uniform((self.p, t)), name=name)
        new_joint = self.create_joint(t)
        opt = tf.optimizers.Adam(learning_rate=learning_rate)

        @tf.function
        def loss():
            log_prob = new_joint.log_prob(dict(
                x = x,
                obs = obs
            ))
            return -log_prob
            
        @tf.function
        def train_step():
            opt.minimize(loss, (x))

        @tf.function
        def recon_loss():
            return tf.reduce_sum((obs - self.vars['w'] @ x) ** 2)

        l = []
        for i in tqdm(range(int(n_iters))):
            train_step()
            l.append(loss())
        return x, np.array(l)


    def setup_tfboard(self):
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = './logs/' + current_time
        return tf.summary.create_file_writer(train_log_dir)





