import numpy as np


class fm:

    def __init__(self, m, num_state_vars, names):
        self.m = m
        self.names = names
        self.num_state_vars = num_state_vars
        self.X0 = np.zeros(self.num_state_vars)
        self.weigths = np.ones(11)
        self.weigths[0] = 0.5
        self.weigths[-1] = 0.5
        self.dt = 1.0 / (10.0)
        self.SetTime()

    def SetTime(self, shift=0):
        self.time = np.linspace(-1.0, self.m - 1 + shift, num=(self.m + shift) + 1, endpoint=True)
        self.nn = len(self.time)
        self.t_quad = np.linspace(-1.0, self.m - 1 + shift, num=10 * (self.m + shift) + 1,
                                  endpoint=True)
        self.n_quad = len(self.t_quad)
        self.result_I = np.zeros(self.nn - 1)
        self.result_D = np.zeros(self.nn - 1)
        self.result_H1 = np.zeros(self.nn - 1)

    def rhs(self, x, t, p):
        pass

    def solve_plain(self, p):
        pass

    def solve(self, p):
        pass

    def ConvertVar(self, v, names):
        return names.index(v)

    def SelectMask(self, v, names, as_col_vec=False):
        v = self.ConvertVar(v, names)
        tmp = np.zeros(len(names))
        tmp[v] = 1
        if as_col_vec:
             return tmp.reshape((len(names), 1))
        else:
             return tmp

