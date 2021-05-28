#coding: utf8

from time import sleep

import os
import sys
import pickle
from scipy import integrate
import scipy.stats as ss
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pandas import datetime, read_excel, read_csv, Timestamp
import pandas as pd
import datetime as dt
import pytwalk
def odeint( rhs, X0, t_quad, args):
    return integrate.odeint(rhs, X0, t_quad, args)
from fm_matrix import fm_matrix


class class_mcmc(fm_matrix):
    def __init__(self, Region, init_day, N, data_fnam, out_fnam, Pobs_I, Pobs_D, init_index,
                 init, init_index0, init0, init_v, exit_probs, R_rates, trim, court, size_window,
                 size_window_new, num_data, delta_day, workdir="./../"):

        self.Region = Region
        self.N = N
        self.N_org = self.N
        self.court = court
        self.delta_day = delta_day
        self.size_window = size_window
        self.size_window_new = size_window_new
        data = np.array(read_csv(data_fnam))
        self.data_all = data
        self.init_day = init_day
        self.Pobs_I = Pobs_I           # probability of recording an infection
        self.Pobs_D = Pobs_D
        self.workdir = workdir
        self.init_v=init_v
        self.init_index = init_index
        self.init_index0 = init_index0
        self.init0 = init0
        self.num_data = num_data
        data = data[self.init_index:self.init_index + self.num_data]
        self.init = init
        self.trim = trim
        if self.trim < 0:
            self.data = data[:self.trim, :]
            self.data_trimed = data[self.trim:, :]
        else:
            self.data = data
            self.data_trimed = 20 + np.zeros((2, 2))

        super().__init__(m=self.data.shape[0], exit_probs=exit_probs, R_rates=R_rates)

        self.Pobs_I = Pobs_I
        self.Pobs_D = Pobs_D
        self.num_betas = 1
        self.solve_plain = self.solve_plain1
        self.out_fnam = out_fnam
        if os.path.isfile(workdir + 'output/' + self.out_fnam + '_samples.pkl'): # check if samples file exists
            print("File with mcmc samples exists, loading samples ...", end=' ')
            self.samples = pickle.load(open(workdir + 'output/' + self.out_fnam + '_samples.pkl', 'rb'))
            self.solns = pickle.load(open(workdir + 'output/' + self.out_fnam + '_solns.pkl', 'rb'))
            self.solns_plain = pickle.load(open(workdir + 'output/' + self.out_fnam + '_solns_plain.pkl', 'rb'))
            self.essize = self.samples.shape[0]
            print("effective sample size: %d" % (self.essize,))
        else:
            print("File with mcmc samples does not exist, run RunMCMC first.")

        if os.path.isfile(workdir + 'output/' + self.out_fnam + '_post_params.pkl'):
            self.post_params = pickle.load(open(workdir + 'output/' + self.out_fnam + '_post_params.pkl', 'rb'))
        else:
            self.post_params={}
        self.workdir = workdir
        self.mcmc_samples = False

    def PrintConfigParameters( self, f=sys.stdout):
        """Prints all configuratiion parameters to file f (defualt sys.stdout)."""
        keys=['Pobs_I', 'Pobs_D', 'f', 'g', 'h', 'i']
        print("\nR_rates:", file=f)
        for key in self.R_rates.keys():
            print("%4s, %7s = %6.4f (%4.1f d), E_list= %2d" %\
                  (key, self.R_rates[key][1], self.R_rates[key][0], 1/self.R_rates[key][0], self.R_rates[key][2]), file=f)
        print("", file=f)
        for key in keys:
            print("%s :" % (key,), file=f)
            print(self._dict_[key], file=f)
            print("", file=f)

    def llikelihood(self, p):
        mu_I, mu_D = self.solve(p)
        mu_I += 3
        mu_D += 3   # the loc parameter is to translate 3
        mu_I *= self.Pobs_I
        mu_D *= self.Pobs_D
        # likelihood for infectious
        omega = 2.0
        theta = 4.0
        r = mu_I/(omega - 1.0 + theta * mu_I)
        q = 1.0/(omega + theta * mu_I)
        log_likelihood_I = np.sum(ss.nbinom.logpmf(self.data[:, 1] + 3, r, q))
        # likelihood for deaths
        omega = 2.0
        theta = 4.0
        r = mu_D / (omega - 1.0 + theta * mu_D)
        q = 1.0/(omega + theta * mu_D)
        log_likelihood_D = np.sum(ss.nbinom.logpmf(self.data[:, 0] + 3, r, q))
        log_likelihood = log_likelihood_D + log_likelihood_I

        return log_likelihood

    # E's A's I's R D beta's(2) omega g
    def lprior(self, p):
        log_prior = 0.0
        if self.court == 0:
            for i in range(self.init_m-2):
                log_prior += ss.gamma.logpdf(p[i], 2.0, scale=10.0)
            log_prior += ss.gamma.logpdf(p[self.mE + self.mIA + self.mIS], 1.0, scale=1.0)   # R
            log_prior += ss.gamma.logpdf(p[self.mE + self.mIA + self.mIS + 1], 1.0, scale=1.0)  # D
            log_prior += np.sum(ss.lognorm.logpdf(p[self.init_m], 1.0, scale=1.0))  # beta
            log_prior += ss.beta.logpdf(p[self.init_m + self.num_betas], 1 + 1 / 6, 1 + 1 / 3)  # omega
            log_prior += ss.beta.logpdf(p[self.init_m + self.num_betas + 1], 1 + 1 / 6, 1 + 1 / 3)  # g
        else:
            for i in range(self.mE):
                a,  scale = self.post_params['E' + str(i) + str(self.court-1)]  # E
                log_prior += ss.gamma.logpdf(p[i], a, scale=scale)
            for i in range(self.mIA):
                a,  scale = self.post_params['IA' + str(i) + str(self.court - 1)]  # IA
                log_prior += ss.gamma.logpdf(p[i + self.mE], a, scale=scale)
            for i in range(self.mIS):
                a, scale = self.post_params['IS' + str(i) + str(self.court - 1)]  # IS
                log_prior += ss.gamma.logpdf(p[i + self.mE + self.mIA], a, scale=scale)

            a, scale = self.post_params['R' + str(self.court - 1)]  # R
            log_prior += ss.gamma.logpdf(p[self.mIS + self.mE + self.mIA], a, scale=scale)

            a, scale = self.post_params['D' + str(self.court - 1)]  # D
            log_prior += ss.gamma.logpdf(p[self.mIS + self.mE + self.mIA + 1], a, scale=scale)

            mu, std = self.post_params['beta' + str(self.court - 1)]  # beta
            log_prior += np.sum(ss.norm.logpdf(p[self.init_m], mu, scale=std))

            a, scale = self.post_params['w' + str(self.court - 1)]
            log_prior += ss.gamma.logpdf(p[self.init_m + self.num_betas], a, scale=scale)   # omega

            a, scale = self.post_params['g' + str(self.court - 1)]
            log_prior += ss.gamma.logpdf(p[self.init_m + self.num_betas + 1], a, scale=scale)   # g

        return log_prior
    
    def energy(self, p):
        return -1 * (self.llikelihood(p) + self.lprior(p))
    
    def support(self, p):
        rt = True
        for i in range(self.init_m):
            rt1 = (0.0 < p[i] < 10.0 ** 8)  # All intial conditions
            rt &= rt1
        rt &= all((0.0 < p[self.init_m:]) * (p[self.init_m:] < 20.0))
        rt &= (0 < p[self.init_m + self.num_betas] < 1.0)   # omega
        rt &= (0 < p[self.init_m + self.num_betas + 1] < 1.0)  # g
        rt &= (p[self.init_m + self.num_betas] * self.N_org - np.sum(p[0: self.init_m]) > 0.0)
        return rt
    
    def sim_init(self):
        """Simulate initial values for mcmc."""
        p = np.zeros(self.num_pars)
        if self.court == 0:
            for i in range(self.init_m-2):
                p[i] = np.random.uniform(low=1, high=10)  # E's A's I's
            p[self.mE + self.mIA + self.mIS] = np.random.uniform(low=0, high=2)      # R
            p[self.mE + self.mIA + self.mIS + 1] = np.random.uniform(low=0, high=2)  # D
            p[self.init_m] = np.random.uniform(low=0.01, high=5.0)   # beta
            p[self.init_m + self.num_betas] = np.random.uniform(0.5, 1)      # w
            p[self.init_m + self.num_betas + 1] = np.random.uniform(0.1, 1)  # g
        else:
            for i in range(self.mE):
                a, scale = self.post_params['E' + str(i) + str(self.court - 1)]  # E
                p[i] = ss.gamma.rvs(a, scale=scale)

            for i in range(self.mIA):
                a, scale = self.post_params['IA' + str(i) + str(self.court - 1)]  # A
                p[i + self.mE] = ss.gamma.rvs(a, scale=scale)

            for i in range(self.mIS):
                a, scale = self.post_params['IS' + str(i) + str(self.court - 1)]  # I
                p[i + self.mE + self.mIA] = ss.gamma.rvs(a, scale=scale)

            a, scale = self.post_params['R' + str(self.court - 1)]  # R
            p[self.mIS + self.mE + self.mIA] = ss.gamma.rvs(a, scale=scale)

            a, scale = self.post_params['D' + str(self.court - 1)]  # D
            p[self.mIS + self.mE + self.mIA + 1] = ss.gamma.rvs(a, scale=scale)

            mu, std = self.post_params['beta' + str(self.court - 1)]  # beta
            p[self.init_m] = ss.norm.rvs(mu, scale=std)

            a, scale = self.post_params['w' + str(self.court - 1)]  # w
            p[self.init_m + self.num_betas] = ss.gamma.rvs(a, scale=scale)

            a, scale = self.post_params['g' + str(self.court - 1)]  # g
            p[self.init_m + self.num_betas + 1] = ss.gamma.rvs(a, scale=scale)

        return p

    def fit_posterior(self):
        scl = 0.99
        sol_tfinal = self.solns_plain[:, self.size_window-1, :]
        for i in range(self.mE):
            mask_E = np.zeros(self.num_state_vars)
            mask_E[self.index_E[i]] = 1
            E = sol_tfinal @ mask_E
            # ------------------
            mu = E.mean()
            std = E.mean() * scl
            scale = std ** 2 / mu
            a = mu / scale
            self.post_params['E' + str(i) + str(self.court)] = (a, scale)

        for i in range(self.mIA):
            mask_IA = np.zeros(self.num_state_vars)
            mask_IA[self.index_IA[i]] = 1
            IA = sol_tfinal @ mask_IA
            mu = IA.mean()
            std = IA.mean() * scl
            scale = std ** 2 / mu
            a = mu / scale
            self.post_params['IA' + str(i) + str(self.court)] = (a, scale)

        for i in range(self.mIS):
            mask_IS = np.zeros(self.num_state_vars)
            mask_IS[self.index_IS[i]] = 1
            IS = sol_tfinal @ mask_IS
            mu = IS.mean()
            std = IS.mean() * scl
            scale = std ** 2 / mu
            a = mu / scale
            self.post_params['IS' + str(i) + str(self.court)] = (a, scale)

        R = sol_tfinal @ self.mask_R
        mu = R.mean()
        std = R.mean() * scl
        scale = std ** 2 / mu
        a = mu / scale
        self.post_params['R' + str(self.court)] = (a, scale)

        D = sol_tfinal @ self.mask_D
        mu = D.mean()
        std = D.mean() * scl
        scale = std ** 2 / mu
        a = mu / scale
        self.post_params['D' + str(self.court)] = (a, scale)

        Beta = self.samples[:, self.init_m]
        self.post_params['beta' + str(self.court)] = ss.norm.fit(Beta)

        w = self.samples[:, self.init_m + self.num_betas]
        mu = w.mean()
        std = w.mean() * scl
        scale = std ** 2 / mu
        a = mu / scale
        self.post_params['w' + str(self.court)] = (a, scale)

        g = self.samples[:, self.init_m + self.num_betas + 1]
        mu = g.mean()
        std = g.mean() * scl
        scale = std ** 2 / mu
        a = mu / scale
        self.post_params['g' + str(self.court)] = (a, scale)


    def RunMCMC(self, T, x0, xp0, burnin=1000, pred=100, plot_fit=True):

        self.SetTime(0)
        self.twalk = pytwalk.pytwalk(n=self.num_pars, U=self.energy, Supp=self.support)
        self.twalk.Run(T=T, x0=x0, xp0=xp0)
        self.mcmc_samples = True
        
        self.iat = int(self.twalk.IAT(start=burnin)[0, 0])
        self.burnin = burnin
        print("\nEffective sample size: %d" % ((T-burnin)/self.iat,))
        self.samples = self.twalk.Output[burnin::(self.iat), :]  # Burn in and thining, output t-walk
        self.essize = self.samples.shape[0]

        self.SetTime(pred)
        # solutions I and D

        self.solns = [np.zeros((self.essize, self.m + pred)) for i in range(2)]   # num dias + pred, save I, D
        self.solns_plain = np.zeros((self.essize, self.m + pred, self.q))         # num dias + pred, save all variables
        #self.solns_initial = np.zeros((self.essize, self.m, self.q))  # num dias + pred, save all variables

        print("Sampling %d model solutions." % (self.essize,))

        for index, m in enumerate(self.samples):
            tmp = list(self.solve(m[:-1]))
            self.solns_plain[index, :, :] = self.soln[10::10, :]
            for i, sl in enumerate(tmp):  # cumulative solutions
                self.solns[i][index, :] = np.cumsum(sl)
            if ((index+1) % 100) == 0:
                print(index+1, end=' ')
        self.fit_posterior()
        print("\nSaving files in ", self.workdir + 'output/' + self.out_fnam + '_*.pkl')
        pickle.dump(self.samples, open(self.workdir + 'output/' + self.out_fnam + 'court_' + str(self.court)+'_samples.pkl', 'wb'))
        pickle.dump(self.solns, open(self.workdir + 'output/' + self.out_fnam+ 'court_' + str(self.court) + '_solns.pkl', 'wb'))
        pickle.dump(self.solns_plain, open(self.workdir + 'output/' + self.out_fnam + 'court_' + str(self.court)+ '_solns_plain.pkl', 'wb'))
        outname_var = self.workdir + 'output/'+ self.out_fnam + '_post_params.pkl'

        with open(outname_var, 'wb') as outfile:
            pickle.dump(self.post_params, outfile)

        if plot_fit:
            self.PlotFit()
    
    def CalculateMAP(self):
        nmap = np.int(np.where(self.samples[:, -1] == self.samples[:, -1].min())[0][0])
        self.pmap = self.samples[nmap, :]
        
    def PlotEvolution(self, pred, ty=0, cumm=False, log=False, ax=None,\
                       csv_fnam=None, q=[10, 25, 50, 75, 90], blue=True, add_MRE=False,\
                       color='red', color_q='black', label='Mediana', right_axis=True, label_cases=True):
        every=1

        if ax == None:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.gca()
        else:
            fig = None

        if pred > self.solns[0].shape[1] - self.m:
            print("Maximum prediction days pred=%d" % (self.solns[0].shape[1] - self.m))
            return
        pred = max(pred, -self.trim)   # Pred should cover the trimmed data
        
        #  Prepare data for comparisons
        if ty == 0:
            data = self.data[:, 1]  # Infected reported
            data_trimed = self.data_trimed[:, 1]
            title = 'cases'
        else:
            data = self.data[:, 0]  # Deaths reported
            data_trimed = self.data_trimed[:, 0]
            title = 'deaths'
        
        if isinstance(self.Pobs_I, float):
            Pobs = (self.Pobs_I, self.Pobs_D)

        else:  # nowcasting, for plotting use only the limit nowcasting proportion, ie ignore the nowcasting
            Pobs = (self.Pobs_I[0], self.Pobs_D[0])

        # Cummulative or prevalence, prepare date to compare with solns
        if cumm:
            prevalence = np.cumsum(data)  # aggregate observed data
            future = prevalence[-1] + np.cumsum(data_trimed)
            solns = Pobs[ty] * self.solns[ty]
            ylabel = 'Accumulated ' + title
            title = 'Accumulated ' + title
        else:
            prevalence = data  # aggregate observed data
            future = data_trimed
            solns = np.diff(np.append(np.zeros((self.solns[ty].shape[0], 1)), Pobs[ty]*self.solns[ty], axis=1), axis=1)
            ylabel = 'Confirmed ' + title
            title = 'Incidence ' + title

        # length and shift for plotting
        length = self.m #- self.init_index
        shift = length + pred
        self.SetTime(pred)

        ### The time frame
        days = mdates.drange(self.init, self.init + dt.timedelta(shift), dt.timedelta(days=1))  # how often do de plot
        #days_pred = mdates.drange( self.init+dt.timedelta(shift), self.init+dt.timedelta(shift_pred), dt.timedelta(days=7)) # how often do de plot    
        self.days = days
        
        ### To save all the data for the plot, len(mexico.days) rows with days
        ### columns: year, month, day, datum, datum_pred, map, q_05, q_25, q_50, q_75, q_95
        ###             0      1   2     3            4    5     6    7     8     9     10     
        sv = -np.ones((len(days), 11))
        for i, day in enumerate(days):
            d = dt.date.fromordinal(int(day))
            sv[i, 0] = d.year
            sv[i, 1] = d.month
            sv[i, 2] = d.day
        # Save data and predicted data
        #sv[:length, 3] = prevalence[self.init_index:]
        sv[:length, 3] = prevalence
        if self.trim < 0:
            sv[length:(length-self.trim), 4] = future

        self.MRE = np.zeros(length)
        # Calculate quantiles
        for i in range(length):
            #sv[i, 6:11] = np.quantile(solns[:, self.init_index + i], q=np.array(q)/100)
            sv[i, 6:11] = np.quantile(solns[:, i], q=np.array(q) / 100)
            #self.MRE[i] = np.mean(np.abs(solns[:, self.init_index + i] - prevalence[self.init_index+i])/(1+prevalence[self.init_index + i]))
            self.MRE[i] = np.mean(np.abs(solns[:, i] - prevalence[i])/(1+prevalence[i]))

        for i in range(length, shift, every):
            #sv[i, 6:11] = np.quantile(solns[:, self.init_index + i], q=np.array(q)/100)
            sv[i, 6:11] = np.quantile(solns[:, i], q=np.array(q) / 100)
        self.PE_solns = solns

        if add_MRE:
            MRE_q = np.quantile(self.MRE, q=[ 0.05, 0.5, 0.95])
            ax.annotate( "MRE: [%5.2f-%5.2f-%5.2f]-%5.2f*" % (MRE_q[0], MRE_q[1], MRE_q[2], np.max(self.MRE)),\
                    (0,1), (10, -10), xycoords='axes fraction',\
                    textcoords='offset points', va='top', fontsize=10)
    
        self.acme = []
        #for d in np.argmax(solns[:, self.init_index:], axis=1):
        for d in np.argmax(solns, axis=1):
            self.acme += [self.init + dt.timedelta(int(d))]
        self.acme_qs = []
        #for a_q in np.quantile(np.argmax(solns[:, self.init_index:], axis=1), q=np.array(q)/100):
        for a_q in np.quantile(np.argmax(solns, axis=1), q=np.array(q) / 100):
            self.acme_qs += [self.init + dt.timedelta(int(a_q))]

        ax.plot( days[:shift], sv[:shift, 8], '-', linewidth=2, color=color, label=label)
        if blue: #Blue shaowed quantiles
            ax.fill_between(days[:shift], sv[:shift, 6], sv[:shift, 10], color='blue', alpha=0.25)
            ax.fill_between(days[:shift], sv[:shift, 7], sv[:shift, 9], color='blue', alpha=0.25)
        else:
            ax.plot( days[:shift], sv[:shift, 6], '--', color=color_q, linewidth=1)
            ax.plot( days[:shift], sv[:shift, 10], '--', color=color_q, linewidth=1)

        if label_cases:
            ax.bar(days[:length], prevalence, color='blue', width=0.5, label='Casos', alpha=0.5)
            ax.plot(days[:length], prevalence, 'bo', markersize=2)
        else:
            ax.bar(days[:length], prevalence, color='blue', markersize=2, width=0.5, alpha=0.5)
            ax.plot(days[:length], prevalence, 'bo')
        if self.trim < 0:
            if label_cases:
                ax.bar(days[length:(length-self.trim)], future, color='grey', width=0.5, alpha=0.1, label='Casos futuros')
                ax.plot(days[length:(length-self.trim)], future, 'k*', markersize=2, label='Casos futuros')
            else:
                ax.bar(days[length:(length-self.trim)], future, color='grey', width=0.5, alpha=0.1, lw=1)
                ax.plot(days[length:(length-self.trim)], future, 'k*', markersize=2)

        ax.set_title(self.Region + ' ' + title)
        ax.legend(loc=0, shadow = True)
        # x-axis
        ax.set_xlabel("Date (day.month)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m'))

        if shift < 190:
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
        else:
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=14))
        ax.tick_params( which='major', axis='x', labelsize=12)#, labelrotation=40)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        # y-axis
        ax.set_ylabel(ylabel)
        ax.set_ylim((0, max( 1.1*np.max(sv[:,-1]), 1.1*np.max(prevalence)) ) ) #Max of upper quantiles
        # Grid
        ax.grid(color='grey', which='both', linestyle='--', linewidth=0.5)
        
        if right_axis and not(log):
            ax_p = ax.twinx()
            y1, y2 = ax.get_ylim()
            ax_p.set_ylim(y1*1e5/self.N, y2*1e5/self.N)
            ax_p.set_ylabel('per 100,000')

        if csv_fnam != None:
            q_str = ', '.join(["q_%02d" % (qunt,) for qunt in q])
            np.savetxt( csv_fnam, sv, delimiter=', ', fmt='%.1f', header="año, mes, dia, datum, datum_pred, map, " + q_str, comments='')
        return ax

    def PlotEvolution1(self, pred, solns, court, ty=0, cumm=False, log=False, ax=None,
                       q=[10, 25, 50, 75, 90], blue=True, color='red', color_q='black',
                       label=True, right_axis=False, plotdata=True):

        init = self.init0 + dt.timedelta(days=court * (self.size_window-1))
        init_index = (init - self.init_day).days

        self.SetTime(pred)
        every = 1
        num_data_all = self.data_all[self.init_index0:].shape[0]
        length = self.m
        shift = length + pred
        data_= self.data_all[init_index: init_index + self.num_data]
        if self.trim < 0:
            data = data_[:self.trim, :]
            data_trimed = data_[self.trim:, :]
        else:
            data = data_
            data_trimed = 20 + np.zeros((2, 2))

        days = mdates.drange(init, init + dt.timedelta(shift), dt.timedelta(days=1))  # how often do de plot
        if (self.init_index0 + shift + court * (self.size_window-1)) <= num_data_all:
            data_all = self.data_all[self.init_index0: self.init_index0 + shift + court * (self.size_window-1)]
            days_all = mdates.drange(self.init0, self.init0 + dt.timedelta(shift + court * (self.size_window-1)), dt.timedelta(days=1))

        else:
            data_all = self.data_all[self.init_index0: self.init_index0 + length + court * (self.size_window - 1)]
            days_all = mdates.drange(self.init0, self.init0 + dt.timedelta(length + court * (self.size_window - 1)),
                                     dt.timedelta(days=1))

        if ax == None:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.gca()
        else:
            fig = None

        if pred > solns[0].shape[1] - self.m:
            print("Maximum prediction days pred=%d" % (solns[0].shape[1] - self.m))
            return
        pred = max(pred, - self.trim)  # Pred should cover the trimmed data

        #  Prepare data for comparisons
        if ty == 0:
            data = data[:, 1]  # Infected reported
            data_trimed = data_trimed[:, 1]
            data_all = data_all[:, 1]
            title = 'cases'
        else:
            data = data[:, 0]  # Deaths reported
            data_trimed = data_trimed[:, 0]
            data_all = data_all[:, 0]
            title = 'deaths'

        Pobs = (self.Pobs_I, self.Pobs_D)

        # Cummulative or prevalence, prepare date to compare with solns
        if cumm:
            prevalence = np.cumsum(data)  # aggregate observed data
            prevalence_all = np.cumsum(data_all)
            future = prevalence[-1] + np.cumsum(data_trimed)
            solns = Pobs[ty] * solns[ty]
            ylabel = 'Accumulated ' + title
            title = 'Accumulated ' + title

        else:
            prevalence = data  # aggregate observed data
            prevalence_all = data_all
            future = data_trimed
            solns = np.diff(np.append(np.zeros((solns[ty].shape[0], 1)), Pobs[ty] * solns[ty], axis=1),
                            axis=1)
            ylabel = 'Confirmed ' + title
            title = 'Incidence ' + title

        # length and shift for plotting

        ### To save all the data for the plot, len(mexico.days) rows with days
        ### columns: year, month, day, datum, datum_pred, map, q_05, q_25, q_50, q_75, q_95
        ###             0      1   2     3            4    5     6    7     8     9     10
        sv = -np.ones((len(days), 11))
        for i, day in enumerate(days):
            d = dt.date.fromordinal(int(day))
            sv[i, 0] = d.year
            sv[i, 1] = d.month
            sv[i, 2] = d.day
        # Save data and predicted data
        # sv[:length, 3] = prevalence[self.init_index:]
        sv[:length, 3] = prevalence
        if self.trim < 0:
            sv[length:(length - self.trim), 4] = future

        MRE = np.zeros(length)
        # Calculate quantiles
        for i in range(length):
            sv[i, 6:11] = np.quantile(solns[:, i], q=np.array(q) / 100)
            MRE[i] = np.mean(np.abs(solns[:, i] - prevalence[i]) / (1 + prevalence[i]))

        for i in range(length, shift, every):
            sv[i, 6:11] = np.quantile(solns[:, i], q=np.array(q) / 100)

        acme = []
        # for d in np.argmax(solns[:, self.init_index:], axis=1):
        for d in np.argmax(solns, axis=1):
            acme += [self.init0 + dt.timedelta(int(d))]
        acme_qs = []
        # for a_q in np.quantile(np.argmax(solns[:, self.init_index:], axis=1), q=np.array(q)/100):
        for a_q in np.quantile(np.argmax(solns, axis=1), q=np.array(q) / 100):
            acme_qs += [self.init0 + dt.timedelta(int(a_q))]

        if label:
            ax.plot(days[:shift], sv[:shift, 8], '-', linewidth=2, color=color, label='Median')
        else:
            ax.plot(days[:shift], sv[:shift, 8], '-', linewidth=2, color=color)

        if blue:  # Blue shaowed quantiles
            ax.fill_between(days[:shift], sv[:shift, 6], sv[:shift, 10], color='blue', alpha=0.25)
            ax.fill_between(days[:shift], sv[:shift, 7], sv[:shift, 9], color='blue', alpha=0.25)
        else:
            ax.plot(days[:shift], sv[:shift, 6], '--', color=color_q, linewidth=1)
            ax.plot(days[:shift], sv[:shift, 10], '--', color=color_q, linewidth=1)

                # Plot data and prediction
        if plotdata:
            # how often do de plot
            if log:
                ax.semilogy(days_all, prevalence_all, 'k*', lw=1, label='Cases')
            else:
                # ax.bar(days_, prevalence_all, color='blue', width=0.5, label='Casos', alpha=0.5)
                ax.plot(days_all, prevalence_all, 'k*', markersize=2)

        ax.set_title(self.Region + ' ' + title)
        ax.legend(loc=0, shadow=True)
        # x-axis
        ax.set_xlabel("Date(day.month)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m'))

        if shift < 190:
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
        else:
            #ax.xaxis.set_major_locator(mdates.DayLocator(interval=14))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=14))
        ax.tick_params(which='major', axis='x', labelsize=12)  # , labelrotation=40)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        # y-axis
        ax.set_ylabel(ylabel)
        #ax.set_ylim((0, max(1.1 * np.max(sv[:, -1]), 1.1 * np.max(prevalence))))  # Max of upper quantiles
        # Grid
        ax.grid(color='grey', which='both', linestyle='--', linewidth=0.5)

        if right_axis and not (log):
            ax_p = ax.twinx()
            y1, y2 = ax.get_ylim()
            ax_p.set_ylim(y1 * 1e5 / self.N, y2 * 1e5 / self.N)
            ax_p.set_ylabel('per 100,000')

        return ax


    def plot_cones(self, pred, ty=0, cumm=False, log=False, ax=None,
                       q=[10, 25, 50, 75, 90], blue=True, color='red', color_q='black',
                       label=True, right_axis=False):
        self.PlotEvolution1(pred=pred, solns=self.solns, court=self.court, ty=ty, cumm=cumm, log=log, ax=ax,
                       q=q, blue=blue, color=color, color_q=color_q, label=label, right_axis=right_axis, plotdata=True)

        for i in range(self.court):
            solns = pickle.load(open(self.workdir + 'output/' + self.out_fnam + 'court_' + str(i) + '_solns.pkl', 'rb'))

            self.PlotEvolution1(pred=pred, solns=solns, court=i, ty=ty, cumm=cumm, log=log, ax=ax,
                                q=q, blue=blue, color=color, color_q=color_q, label=False, right_axis=right_axis,
                                plotdata=False)

    def Plot_evol_params_beta(self, court, ax, pred, label, index, q=[10, 25, 50, 75, 90],vacc=False, color='red'):
        workdir = './../'
        samples = pickle.load(open(workdir + 'output/' + self.out_fnam + 'court_' + str(court) + '_samples.pkl', 'rb'))
        beta = samples[:, self.init_m + index]
        init = self.init0 + dt.timedelta(days=court * (self.size_window - 1))
        init_index = (init - self.init_day).days
        every = 1
        length = self.num_data
        shift = length + pred

        if ax == None:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.gca()
        else:
            fig = None
        days = mdates.drange(init, init + dt.timedelta(shift), dt.timedelta(days=1))  # how often do de plot
        sv = -np.ones((len(days), 9))
        for i, day in enumerate(days):
            d = dt.date.fromordinal(int(day))
            sv[i, 0] = d.year
            sv[i, 1] = d.month
            sv[i, 2] = d.day

        for i in range(length):
            sv[i, 4:9] = np.quantile(beta, q=np.array(q) / 100)

        for i in range(length, shift, every):
            sv[i, 4:9] = np.quantile(beta, q=np.array(q) / 100)
        if label == True:
            ax.plot(days[:shift], sv[:shift, 6], '-', linewidth=2, color=color, label=r'Median')
            ax.legend(loc=0, shadow=True)
            ax.fill_between(days[:shift], sv[:shift, 4], sv[:shift, 8], color='blue', alpha=0.25)
            ax.fill_between(days[:shift], sv[:shift, 5], sv[:shift, 7], color='blue', alpha=0.25)
        else:
            ax.plot(days[:self.size_window], sv[:self.size_window, 6], '-', linewidth=2, color=color)
            ax.fill_between(days[:self.size_window], sv[:self.size_window, 4], sv[:self.size_window, 8], color='blue', alpha=0.25)
            ax.fill_between(days[:self.size_window], sv[:self.size_window, 5], sv[:self.size_window, 7], color='blue', alpha=0.25)

        if index == 0:
            ax.set_ylabel(r" $\beta$", fontsize=18)
            ax.set_title(r"%s $(\beta) $" % self.Region)
        elif index == 1:
            ax.set_ylabel(r" $\omega$", fontsize=18)
            ax.set_title(r"%s $(\omega)$" % self.Region)
        else:
            ax.set_ylabel(r" $g$", fontsize=18)
            ax.set_title(r"%s $(g)$" % self.Region)

        ymax = np.max(sv[:, 4:9])
        ax.set_xlabel("Date (day.month)")

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=14))

        if vacc:
            ax.vlines(self.init_v, ymin=0, ymax=ymax, color='k', linestyles='dashed', label='Vaccination')
            ax.vlines(self.init_v + dt.timedelta(21), color='g', ymin=0, ymax=ymax, linestyles='dashed',
                      label='21 days later')
            #ax.set_ylim(0, ymax + ymax * 0.1)
        if label:
            ax.set_xlim(self.init0 - dt.timedelta(1), init + dt.timedelta(shift))
            ax.legend()
        else:
            ax.set_xlim(self.init0 - dt.timedelta(1), init + dt.timedelta(self.size_window))

        ax.tick_params(which='major', axis='x', labelsize=12)  # , labelrotation=40)
        plt.setp(ax.get_xticklabels(), rotation=60, ha="right", rotation_mode="anchor")
        ax.grid(color='grey', which='both', linestyle='--', linewidth=0.5)

        return ax

