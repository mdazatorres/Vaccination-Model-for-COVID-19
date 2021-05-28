#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 01:13:36 2020

@author: jac
"""


import numpy as np
from scipy import linalg as lg
from scipy import integrate
from covid_fm import fm, AuxMatrix

def odeint( rhs, X0, t_quad, args):
    return integrate.odeint(rhs, X0, t_quad, args)


def ModeloSEIRD(m_list, e_list, f, g, prn=False):
    T = AuxMatrix(names="S E I^A I^S R D", prn=prn)
    T.BaseVar('S')
    T.Exit('S', 'E')
    T.SplitExit('E', 'I^A', 'I^S', 1- f, prob_symb=['1-f', 'f'])
    T.Exit('I^A', 'R')
    T.SplitExit('I^S', 'D', 'R', g, prob_symb=['g', '1-g'])
    T.NoExit('R')
    T.NoExit('D')
    T.End()

    # Split in Erlang series of length m
    T.SplitErlang(e_list, m_list)
    return T


class fm_matrix(fm):
  
    def __init__(self, m, exit_probs, R_rates):
        f, g = exit_probs
        self.f = f           # fraction of severe infections
        self.g = g           # fraction of severe infections that require hospitalization
        self.factor_foi = 0.2
        self.R_rates = R_rates  # dictionary of residence rates,
        self.mE = self.R_rates['E'][-1]
        self.mIA = self.R_rates['I^A'][-1]
        self.mIS = self.R_rates['I^S'][-1]
        self.mR = 1
        self.mD = 1
        # number of the initial conditions to estimate
        self.init_m = self.mE + self.mIA + self.mIS + self.mR + self.mD
        self.num_pars = self.init_m + 3     # E's A's I's R D beta omega g

        m_list = [self.R_rates[v][2] for v in self.R_rates.keys()]
        e_list = list(self.R_rates.keys())

        # Define the graph matrix describing the model (see above)
        self.T = ModeloSEIRD(m_list, e_list, f=f, g=g, prn=False)
        self.n = self.T.n  # original number of state variables
        self.q = self.T.q  # Total number of state variables

        super().__init__(m, num_state_vars=self.q)  # to call base class fm
        #self.Wt = np.random.normal(0, 1, len(self.time))
        # "S E I^A I^S R D"
        self.par = np.zeros(self.n)  # Par mask for original state variables
        self.R_rates = R_rates
        # Known parameters, set residence rates:
        for v in R_rates.keys():
            self.par[self.T.ConvertVar(v)] = R_rates[v][0]

        # Auxiliars
        self.sigma_1 = self.par[self.T.ConvertVar('E')]
        self.sigma_2 = self.par[self.T.ConvertVar('I^S')]
        self.gamma_1 = self.par[self.T.ConvertVar('I^A')]

        # The masks to select variables from list of state variables
        self.mask_S = self.T.SelectMask('S')
        self.mask_E = self.T.SelectMask('E')
        self.mask_R = self.T.SelectMask('R')
        self.mask_D = self.T.SelectMask('D')
        self.mask_Es = self.T.SelectMask('E', E_range='all', as_col_vec=True)
        self.mask_Ess = self.T.SelectMask('E', E_range='all')
        self.mask_IA = self.T.SelectMask('I^A')
        self.mask_IAs = self.T.SelectMask('I^A', E_range='all')
        self.mask_ISs = self.T.SelectMask('I^S', E_range='all', as_col_vec=True)
        self.mask_ISs_flat = self.T.SelectMask('I^S', E_range='all')
        self.mask_IS = self.T.SelectMask('I^S')
        self.index_E = np.where(self.mask_Ess)[0]
        self.index_IA = np.where(self.mask_IAs)[0]
        self.index_IS = np.where(self.mask_ISs_flat)[0]
        self.X0 = np.zeros((self.q, ))
                
    def GetMask(self, v, E_range='all', as_col_vec=False):
        return self.T.SelectMask(v, E_range=E_range, as_col_vec=as_col_vec)

    def rhs(self, x, t, p):
        #beta = p[self.init_m + np.where(t < self.intervention_day)[0][0]]
        beta = p[self.init_m]
        I_A = np.sum(x * self.mask_IAs)      # total number of asymptomatic infections
        I_S = np.sum(x * self.mask_ISs_flat)  # total number of asymptomatic infections

        # force of infection beta1*I^A/N + some factor of  beta1*I^S/N
        foi = (I_A + self.factor_foi * I_S)/self.N * beta
        self.par[self.T.ConvertVar('S')] = foi
        self.T.M[-1][-3] = p[self.init_m + self.num_betas + 1]  # g
        self.T.M[-2][-3] = 1 - p[self.init_m + self.num_betas + 1]  # 1-g
        return self.T.M @ (x * (self.T.par_mask @ self.par))

    def solve_plain(self, p, quad=True):
        pass
        
    def solve_plain1(self, p, quad=True):
        self.N = p[self.init_m + self.num_betas] * self.N_org
        aux_mask_Ess = np.copy(self.mask_Ess)
        aux_mask_IAs = np.copy(self.mask_IAs)
        aux_mask_ISs = np.copy(self.mask_ISs_flat)

        aux_mask_Ess[self.mask_Ess == 1] = p[0: self.mE]
        aux_mask_IAs[self.mask_IAs == 1] = p[self.mE: self.mE + self.mIA]
        aux_mask_ISs[self.mask_ISs_flat == 1] = p[self.mE + self.mIA: self.mE + self.mIA + self.mIS]

        self.X0 *= 0
        self.X0 += aux_mask_Ess
        self.X0 += aux_mask_IAs
        self.X0 += aux_mask_ISs
        self.X0 += p[self.mE + self.mIA + self.mIS] * self.mask_R
        self.X0 += p[self.mE + self.mIA + self.mIS + 1] * self.mask_D
        self.X0 += (self.N - np.sum(p[0: self.init_m])) * self.mask_S
        if quad:
            return odeint(self.rhs, self.X0, self.t_quad, args=(p,))
        else:
            return odeint(self.rhs, self.X0, self.time, args=(p,))


    def solve(self, p):
        self.soln = self.solve_plain(p)
        for k in range(0, self.nn-1):
            # x_s = np.sum(self.soln[(10 * k): (10 * (k + 1) + 1), :] @ self.mask_ISs, axis=1)
            # incidence = self.g * self.sigma_2 * x_s
            x_e = np.sum(self.soln[(10 * k):(10 * (k + 1) + 1), :] @ self.mask_Es, axis=1)
            incidence = self.f * self.sigma_1 * x_e
            self.result_H1[k] = self.dt * np.dot(self.weigths, incidence)
        return self.result_H1, np.diff(self.soln[::10, :] @ self.mask_D)


