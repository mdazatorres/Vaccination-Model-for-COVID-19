import numpy as np
from scipy import linalg as lg
from scipy import integrate
from covid_fm_v import fm
from pandas import datetime, read_excel, read_csv, Timestamp

def odeint( rhs, X0, t_quad, args):
    return integrate.odeint(rhs, X0, t_quad, args)

class fm_matrix_v(fm):
  
    def __init__(self, m, N, lamda_v1, lamda_v2):
        self.N = N
        self.workdir = "./../"
        self.N_org = self.N
        self.names = ['S', 'V_1', 'V_2', 'E', 'U', 'O', 'R', 'D']
        self.gamma_o = 1/14
        self.gamma_u = 1/7
        self.eps = 0.95
        self.sigma = 1/5
        self.f = 0.6
        self.lamda_v1 = lamda_v1
        self.lamda_v2 = lamda_v2
        self.epsilon = 0.95
        self.q = len(self.names)
        self.X0 = np.zeros(self.q)
        super().__init__(m, num_state_vars=self.q, names=self.names)
        self.mask_E = self.SelectMask('E', names=self.names, as_col_vec=True)
        self.mask_D = self.SelectMask('D', names=self.names, as_col_vec=False)
        self.mask_V1 = self.SelectMask('V_1', names=self.names, as_col_vec=False)
        self.mask_V2 = self.SelectMask('V_2', names=self.names, as_col_vec=False)
        self.factor_foi = 0.2

    def rhs(self, x, t, p):
        beta = p[self.q]
        g = p[self.q + 2]
        fx = np.zeros(self.q)
        self.lamda = beta * (x[4] + x[5] * self.factor_foi)/self.N
        fx[0] = - self.lamda * x[0] - self.lamda_v1 * self.epsilon * x[0]
        fx[1] = self.lamda_v1 * self.epsilon * (x[0] + x[3] + x[4] + x[6]) - self.lamda_v2 * x[1] - self.lamda * x[1] * 0.4 # V1
        fx[2] = self.lamda_v2 * x[1] - self.lamda * x[2] * 0.05  # V2
        fx[3] = self.lamda * x[0] - self.sigma * x[3] - self.lamda_v1 * self.epsilon * x[3] + self.lamda * x[2] * 0.05 + self.lamda * x[1] * 0.4  # E
        fx[4] = (1 - self.f) * self.sigma * x[3] - self.gamma_u * x[4] - self.lamda_v1 * self.epsilon * x[4]  # U
        fx[5] = self.f * self.sigma * x[3] - self.gamma_o * x[5]   # O
        fx[6] = (1 - g) * self.gamma_o * x[5] + self.gamma_u * x[4] - self.lamda_v1 * self.epsilon * x[6]  # R
        fx[7] = g * self.gamma_o * x[5]     # D
        return fx

    def solve_plain(self, p, quad=True):

        self.N = p[self.q+1] * self.N_org
        self.X0 = p[:self.q]
        if quad:
            return odeint(self.rhs, self.X0, self.t_quad, args=(p,))
        else:
            return odeint(self.rhs, self.X0, self.time, args=(p,))

    def solve(self, p):
        self.soln = self.solve_plain(p)
        for k in range(0, self.nn-1):
            x_e = np.sum(self.soln[(10 * k):(10 * (k + 1) + 1), :] @ self.mask_E, axis=1)
            incidence = self.f * self.sigma * x_e
            self.result_H1[k] = self.dt * np.dot(self.weigths, incidence)
        return self.result_H1, np.diff(self.soln[::10, :] @ self.mask_D)

