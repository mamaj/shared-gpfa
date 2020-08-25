import datetime
import math
import warnings

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tqdm.auto import tqdm, trange
from sklearn.decomposition import FactorAnalysis
import matplotlib.pyplot as plt

tfd = tfp.distributions
tfb = tfp.bijectors
dtype = np.float32


class SharedGpfa:

    def __init__(self, m, q, p, latent_noise_var=1e-4, dtype=np.float32, name="SGPFA", color='crimson'):
        self.m = m
        self.q = q
        self.p = p
        self.t = None
        self.latent_noise_var = latent_noise_var
        self.amplitude = math.sqrt(1 - latent_noise_var)
        self.dtype = dtype
        self.vars = dict()
        self.joint = []
        self.name = name
        self.color = color

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
            name='w')
        self.vars['length_scale'] = tfp.util.TransformedVariable(
            init_log((self.p)),
            constrain_positive,
            name='length_scale')
        self.vars['subject_noise_scale'] = tfp.util.TransformedVariable(
            init_log(self.m),
            constrain_positive,
            name='subject_noise_scale')
        self.vars['roi_noise_scale'] = tfp.util.TransformedVariable(
            init_log(self.q),
            constrain_positive,
            name='roi_noise_scale')
        self.vars['x'] = []
        for i, t in enumerate(self.t):
            x = tf.Variable(init_uniform((self.p, t)), name=f'x_{i}')
            self.vars['x'].append(x)
            
    def init_joint(self):
        self.check_model_fit()
        self.joint.clear()
        for t in self.t:
            self.joint.append(self.create_joint(t))
            
    def create_joint(self, t):
        ind_points = np.arange(t).astype(self.dtype)
        joint = tfd.JointDistributionNamed(dict(
            x=tfd.Independent(
                tfd.GaussianProcess(
                    kernel=tfp.math.psd_kernels.ExponentiatedQuadratic(
                        amplitude=self.amplitude,
                        length_scale=self.vars['length_scale'],
                        feature_ndims=1),
                    index_points=ind_points[:, None],
                    jitter=1e-5,
                    observation_noise_variance=self.latent_noise_var,
                    validate_args=True),
                reinterpreted_batch_ndims=1),
            
            w=tfd.Independent(
                tfd.Normal(
                    loc=np.zeros((self.m, self.q, self.p), dtype=self.dtype),
                    scale=1),
                reinterpreted_batch_ndims=3),
            
            obs=lambda x, w:
                tfd.Independent(
                    tfd.Normal(
                        loc=tf.matmul(w, tf.expand_dims(x, -3)),
                        scale=tf.expand_dims(tf.expand_dims(
                            self.vars['subject_noise_scale'], -1) * self.vars['roi_noise_scale'], -1)),
                reinterpreted_batch_ndims=3)
        ))
        return joint
    
    def create_joint_subjects(self, t, subs):
        if subs is Ellipsis:
            return self.create_joint(t)
        elif type(subs) is int:
            subs = [subs]

        ind_points = np.arange(t).astype(self.dtype)
        joint = tfd.JointDistributionNamed(dict(
            x=tfd.Independent(
                tfd.GaussianProcess(
                    kernel=tfp.math.psd_kernels.ExponentiatedQuadratic(
                        amplitude=self.amplitude,
                        length_scale=self.vars['length_scale'],
                        feature_ndims=1),
                    index_points=ind_points[:, None],
                    jitter=1e-5,
                    observation_noise_variance=self.latent_noise_var,
                    validate_args=True),
                reinterpreted_batch_ndims=1),
            
            w=tfd.Independent(
                tfd.Normal(
                    loc=np.zeros((len(subs), self.q, self.p), dtype=self.dtype),
                    scale=1),
                reinterpreted_batch_ndims=3),
            
            obs=lambda x, w:
                tfd.Independent(tfd.Normal(
                    loc=tf.matmul(w, tf.expand_dims(x, -3)),
                    scale=tf.expand_dims(tf.expand_dims(
                        tf.gather(self.vars['subject_noise_scale'], subs), -1) * self.vars['roi_noise_scale'], -1)
                ), 3),
        ))
        return joint

    def log_prob(self, data, joint=None, x=None, w=None):
        self.check_model_fit()
        if joint is None:
            joint = self.joint
        if x is None:
            x = self.vars['x']
        if w is None:
            w = self.vars['w']
        data = self.add_batch(data)
        x = self.add_batch(x)
        
        log_prob = []
        for _joint, _x, _data in zip(joint, x, data):
            log_prob.append(
                _joint.log_prob(dict(
                    x=_x,
                    w=w,
                    obs=_data
                )))
        return tf.reduce_sum(log_prob)
    
    def log_prob_parts(self, data, joint=None, x=None, w=None):
        self.check_model_fit()
        if joint is None:
            joint = self.joint
        if x is None:
            x = self.vars['x']
        if w is None:
            w = self.vars['w']
        data = self.add_batch(data)
        x = self.add_batch(x)
        
        log_prob_x = []
#         log_prob_w = []
        log_prob_obs = []
        for _joint, _x, _data in zip(joint, x, data):
            log_prob = _joint.log_prob_parts(dict(
                x=_x,
                w=w,
                obs=_data))
            log_prob_x.append(log_prob['x'])
#             log_prob_w.append(log_prob['w'])
            log_prob_obs.append(log_prob['obs'])
            
        return {'x': tf.reduce_sum(log_prob_x),
                'w': log_prob['w'],
                'obs': tf.reduce_sum(log_prob_obs)}

    def train_loss(self, data, reg, smoothness, joint=None, x=None, w=None, normalize=True):
        log_prob = self.log_prob_parts(data, joint, x, w)
        loss = log_prob['obs'] + smoothness * log_prob['x'] + reg * log_prob['w']
        if normalize:
            m = data[0].shape[0]
            t = [d.shape[-1] for d in data]
            normalizer = sum(t) * (self.p + self.q * m)
        else:
            normalizer = 1
        return -loss / normalizer

    def fit(self, train_data, n_iters, learning_rate=.01, tensorboard=False, fa_init=True, smoothness=1, reg=1, sm_factor=None, desc='fitting SGPFA', **kwargs):
        if sm_factor is not None:
            smoothness = (self.m * self.q / self.p) / sm_factor
            print(smoothness)
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
            return self.train_loss(train_data, smoothness=smoothness, reg=reg)

        @tf.function
        def train_step():
            opt.minimize(loss, trainable_vars)

        l = []
        for epoch in tqdm(range(int(n_iters)), desc=desc, **kwargs):
            train_step()
            l.append(loss())
            if tensorboard:
                self.update_tfsummary(loss(), epoch)
        return l

    def add_video(self, obs, smoothness=1, sm_factor=None, n_iters=1e3, learning_rate=0.04, ls_init=True, name='new_x', tensorboard=False, desc='Adding Video', **kwargs):

        if sm_factor is not None:
            smoothness = (self.m * self.q / self.p) / sm_factor
            
        obs = self.add_batch(obs)
#         assert obs[0].shape[:-1] == (self.m, self.q), f"obs: {obs[0].shape}, (m = {self.m}, q = {self.q}))"

        t = [x.shape[-1] for x in obs]
        new_joint = [self.create_joint(_t) for _t in t]

        if ls_init:
            xlist_pinv = self.least_square_traj(obs, subs=Ellipsis)
        else:
            def init(size):
                return np.random.uniform(size=size).astype(self.dtype)

        xlist = []
        for i, _t in enumerate(t):
            if ls_init:
                x_init = xlist_pinv[i]
            else:
                x_init = init((self.p, _t))
            x = tf.Variable(x_init, name=name + f'_{i}', dtype=self.dtype)
            xlist.append(x)

        if tensorboard:
            self.setup_tfboard()
        opt = tf.optimizers.Adam(learning_rate=learning_rate)

        @tf.function
        def loss():
            return self.train_loss(obs, reg=1, smoothness=smoothness, joint=new_joint, x=xlist)

        @tf.function
        def train_step():
            opt.minimize(loss, xlist)

        for epoch in trange(int(n_iters), desc=desc, **kwargs):
            train_step()
            if tensorboard:
                self.update_tfsummary(loss(), epoch)
                with self.summary_writer.as_default():
                    for i, x in enumerate(xlist):
                        tf.summary.histogram(f'xlist{i}_0', x[0], step=epoch)
                        tf.summary.histogram(f'xlist{i}_1', x[1], step=epoch)
        return xlist

    def add_video_subjects(self, obs, smoothness=1, sm_factor=None, n_iters=1e3, learning_rate=0.04, ls_init=True, subs=Ellipsis, name='new_x', tensorboard=False, desc='Adding Video', **kwargs):

        if type(subs) is int:
            subs = [subs]
        obs = self.add_batch(obs)
        assert len(subs) == len(obs[0])
        
        if sm_factor is not None:
            smoothness = (len(subs) * self.q / self.p) / sm_factor


        t = [x.shape[-1] for x in obs]
        new_joint = [self.create_joint_subjects(_t, subs=subs) for _t in t]

        if ls_init:
            xlist_pinv = self.least_square_traj(obs, subs)
        else:
            def init(size):
                return np.random.normal(size=size).astype(self.dtype)

        xlist = []
        for i, _t in enumerate(t):
            if ls_init:
                x_init = xlist_pinv[i]
            else:
                x_init = init((self.p, _t))
            x = tf.Variable(x_init, name=name + f'_{i}', dtype=self.dtype)
            xlist.append(x)

        if tensorboard:
            self.setup_tfboard()
        opt = tf.optimizers.Adam(learning_rate=learning_rate)

        @tf.function
        def loss():
            return self.train_loss(obs, reg=1, smoothness=smoothness, joint=new_joint, x=xlist, w=self.vars['w'][subs])

        @tf.function
        def train_step():
            opt.minimize(loss, xlist)

        for epoch in trange(int(n_iters), desc=desc, **kwargs):
            train_step()
            if tensorboard:
                self.update_tfsummary(loss(), epoch)
                with self.summary_writer.as_default():
                    for i, x in enumerate(xlist):
                        tf.summary.histogram(f'xlist{i}_0', x[0], step=epoch)
                        tf.summary.histogram(f'xlist{i}_1', x[1], step=epoch)
        return xlist

    def add_subject(self, obs, video=0, reg=1, solve_ls=True, add_to_model=False, n_iters=1e3, learning_rate=0.04, name='w_new'):
        if obs.ndim == 2:
            obs = np.expand_dims(obs, 0)

        if solve_ls:
            # w = obs @ np.linalg.pinv(self.vars['x'][video].numpy())
            X = self.traj(video)
            w = obs @ X.T @ np.linalg.inv((X @ X.T + reg * np.eye(self.p)))

        else:
            return None
            # def init_norm(size):
            #     return np.random.normal(size=size).astype(self.dtype)
            # w = tf.Variable(init_norm((obs.shape[0], self.q, self.p)), name=name)
            # opt = tf.optimizers.Adam(learning_rate=learning_rate)

            # @tf.function
            # def loss():
            #     return tf.reduce_sum((obs - w @ self.vars['x'][video]) ** 2)

            # @tf.function
            # def train_step():
            #     opt.minimize(loss, (w))

            # for i in range(int(n_iters)):
            #     train_step()
        
        if add_to_model:
            self.vars['w'] = tf.concat((self.vars['w'], w), axis=0, name='w_concat')
            self.vars['subject_noise_scale'] = tf.concat((self.vars['subject_noise_scale'], [1]), axis=0, name='subject_noise_scale_concat')
            self.m += 1
            self.init_joint()
        return w
    
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

    def least_square_traj(self, data, subs):
        data = self.add_batch(data)
        w = self.vars['w'].numpy()[subs]
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
            )
            for v in transformed_vars:
                trainable_vars.append(self.vars[v].trainable_variables[0])
        return trainable_vars

    def check_model_fit(self):
        assert self.t is not None, ".fit() has not been called yet."

    def add_batch(self, data):
        if type(data) is not list and data.ndim == 2:
            data = [np.expand_dims(data, axis=0)]
        if type(data) is not list and data.ndim == 3:
            return [data]
        else:
            return data

    def len_scale(self, unit='TR', tr_to_sec=2):
        tau_tr = tf.convert_to_tensor(self.vars['length_scale']).numpy()
        if unit == 'TR':
            return tau_tr
        elif unit == 's':
            return tau_tr * tr_to_sec
        elif unit == 'ms':
            return tau_tr * tr_to_sec * 1000
        elif unit == 'hz':
            return 1 / (tau_tr * tr_to_sec)
        else:
            return None
        
        
    def sub_noise(self):
        return tf.convert_to_tensor(self.vars['subject_noise_scale']).numpy()
    
    def reg_noise(self):
        return tf.convert_to_tensor(self.vars['roi_noise_scale']).numpy()
    
    def weights(self):
        return self.vars['w'].numpy()
    
    def traj(self, exp=0):
        if exp == "all":
            return np.concatenate([x.numpy() for x in self.vars['x']], axis=-1)
        else:
            return self.vars['x'][exp].numpy()
    
    def kernel_logdet(self):
        cov = self.joint[0].submodules[3].covariance().numpy()
        return np.linalg.slogdet(cov)[1]
    
    def evidence_laplace(self, data):
        # P(D) ~= logP(D,\theta) + (M/2) log(2\pi) -1/2 log(det(H))
        # H is the Hessian of -logP(D,\theta) w.r.t \theta
        with tf.GradientTape(persistent=True) as tape:
            grads = tape.gradient(-self.log_prob(data), self.get_trainable_vars())
            flattened_grads = tf.concat([tf.reshape(grad, [-1]) for grad in grads], axis=0)
        hessians = tape.jacobian(flattened_grads, self.get_trainable_vars())
        flattened_hessians = tf.concat([tf.reshape(hess, [hess.shape[0], -1]) for hess in hessians], 1)
        logdet_hess = np.linalg.slogdet(flattened_hessians)
#         assert logdet_hess[0] > 0
        return (self.log_prob(data).numpy()
                + flattened_hessians.shape[0] * np.log(2 * np.pi) / 2
                + logdet_hess[1] / 2)

    def modified_bic(self, data):
        # P(D) ~= logP(D,\theta) + (M/2) log(2\pi) -1/2 log(det(N * H_hat))
        # P(D) ~= logP(D,\theta) - (M/2) log(N / 2\pi)
        data = self.add_batch(data)
        N = len(data)
        dof = sum(tf.size(v).numpy() for v in self.get_trainable_vars())
        return self.log_prob(data).numpy() - dof * np.log(N / (2 * np.pi)) / 2
    
    def bic_joint(self, data):
        # P(D) ~= logP(D,\theta) - (M/2) log(N)
        data = self.add_batch(data)
        N = len(data)
        dof = sum(tf.size(v).numpy() for v in self.get_trainable_vars())
        return self.log_prob(data).numpy() - dof * np.log(N) / 2

    def bic(self, data):
        # P(D) ~= logP(D|\theta) - (M/2) log(N)
        data = self.add_batch(data)
        N = len(data)
        dof = sum(tf.size(v).numpy() for v in self.get_trainable_vars())
        return self.log_prob_parts(data)['obs'].numpy() - dof * np.log(N) / 2
    
    def aic(self, data):
        dof = sum(tf.size(v).numpy() for v in self.get_trainable_vars())
        return self.log_prob(data).numpy() - dof
    
    def sort(self, idx=None, sgn=None):
        if idx is None:
            idx = self.len_scale().argsort()
                
        new_lenscale = self.len_scale()[idx]
        new_w = self.weights()[..., idx]
        self.vars['length_scale'].assign(new_lenscale)
        self.vars['w'].assign(new_w)
        for i, x in enumerate(self.vars['x']):
            new_x = self.traj(exp=i)[idx]
            x.assign(new_x)
            
        if sgn is not None:
            sgn = np.sign(sgn)
            self.vars['w'].assign(self.weights() * sgn)
            for i, x in enumerate(self.vars['x']):
                new_x = self.traj(exp=i) * sgn.reshape(-1, 1)
                x.assign(new_x)

        
    def vars_summary(self, tablen=30):
        print('Name \t Shape \t Elements'.expandtabs(tablen))
        print((('-' * 10 + '\t') * 3).expandtabs(tablen))
        for var in self.get_trainable_vars():
            print(f'{var.name} \t {var.shape} \t {tf.size(var).numpy()}'.expandtabs(30))
        print(' \t \t '.expandtabs(tablen) + '-' * 10)
        num_vars = sum(tf.size(v).numpy() for v in self.get_trainable_vars())
        print(f' \t \t {num_vars}'.expandtabs(tablen))

    
    def plot_parameters(self, figsize=(18, 4), color='r'):
        fig, axs = plt.subplots(1, 4, figsize=figsize, constrained_layout=True)
        fig.suptitle('Parameters', fontsize=15)
    
        axs[0].set_title('Length Scale')
        axs[0].bar(np.arange(self.p), self.len_scale(), alpha=0.4, label=self.name, color=color)
        axs[0].set_xlabel('Latent Dimension')
        axs[0].legend(loc=2)
        axs[0].set_xticks(np.arange(self.p))
        axs[0].set_xticklabels([fr'$\tau_{i}$' for i in range(self.p)])

        axs[1].set_title('subject noise scale')
        axs[1].plot(self.sub_noise(), label=self.name, color=color)
        axs[1].set_xlabel('subjects')
        axs[1].legend(loc=2)
        axs[1].set_xticks(np.arange(self.m))
        axs[1].set_xticklabels(np.arange(self.m))

        axs[2].set_title('Regions noise scale')
        axs[2].plot(self.reg_noise(), label=self.m, color=color)
        axs[2].set_xlabel('Regions')
        axs[2].legend(loc=2)
        axs[2].set_xticks(np.arange(self.q))
        axs[2].set_xticklabels(np.arange(self.q))

        axs[3].set_title('overal observation noise scale')
        noise = self.sub_noise()[:, None] * self.reg_noise()
        axs[3].plot(noise.reshape(-1), label=self.name, color=color)
        axs[3].set_xlabel('Subjects/Regions')
        axs[3].legend(loc=2)
        axs[3].set_xticks(np.arange(0, self.m * self.q, self.q))
        axs[3].set_xticks(np.arange(self.m * self.q), minor=True)
        axs[3].set_xticklabels(np.arange(self.m))