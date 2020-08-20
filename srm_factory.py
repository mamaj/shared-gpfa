import numpy as np
import brainiak.funcalign.srm

def SRM(*args, **kwargs):
    return srm_factory(brainiak.funcalign.srm.SRM)(*args, name='SSRM', **kwargs)

def RSRM(*args, **kwargs):
    return srm_factory(brainiak.funcalign.srm.RSRM)(*args, name='RSRM', **kwargs)

def srm_factory(base):
    
    class SRMwrapper(base):
        """ A simple wrapper for SRM class."""

        def data_decorator(wrapped):
            def wrapper(self, X, *args, **kwargs):
                if type(X) is list and X[0].ndim == 3:
                    X = np.concatenate(X, axis=-1)
                X = [_x for _x in X]
                return wrapped(self, X, *args, **kwargs)
            return wrapper

        def __init__(self, name="SRM", color='seagrean', *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.name = name
            self.color = color

        @data_decorator
        def fit(self, *args, **kwargs):
            super().fit(*args, **kwargs)

        @data_decorator
        def transform(self, *args, **kwargs):
            xlist = super().transform(*args, **kwargs)
            return np.array(xlist)

        def sub_noise(self):
            return self.rho2_

        def weights(self):
            return np.array(self.w_)

        def traj(self):
            return self.s_

        def len_scale(self):
            return np.full((self.s_.shape[0]), np.nan)

        def sort(self, idx=None, sgn=None):
            if idx is None:
                idx = np.diag(self.sigma_s_).argsort()

            self.sigma_s_ = self.sigma_s_[np.ix_(idx, idx)]
            self.s_ = self.s_[idx, :]
            for i in range(len(self.w_)):
                self.w_[i] = self.w_[i][:, idx]

            if sgn is not None:
                sgn = np.sign(sgn)
                self.s_ *= sgn.reshape(-1, 1)
                for i in range(len(self.w_)):
                    self.w_[i] *= sgn

    return SRMwrapper