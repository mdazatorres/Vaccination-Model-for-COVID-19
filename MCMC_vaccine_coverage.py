import numpy as np
import matplotlib.pyplot as plt
from pytwalk import pytwalk
import pandas as pd
import scipy.stats as ss
from scipy.stats import gamma
import matplotlib.ticker as ticker
plt.rcParams['font.size'] = 35
# Data
N= 39510000
n = 30
data = pd.read_excel('CA-Vaccine.xlsx')
data = data[data['county']=='All CA Counties']
dates = data['administered_date'][-n:]
A_data = np.array(data['cumulative_at_least_one_dose'])[-n:]
U_data = np.array(data['cumulative_fully_vaccinated'])[-n:]
V_data = A_data - U_data
W_data = N - A_data

t = np.linspace(0,n-1,n)
U0 = U_data[0]
A0 = A_data[0]
W0 = N - A0
V0 = V_data[0]
X0 = [W0, V0, U0, A0]  # intial condition

alp_l1 = 3
bet_l1 = 10
alp_l2 = 3
bet_l2 = 10

def F(p):
    l1 = p[0]
    l2 = p[1]
    C = (l1 * W0 / (l1 - l2)) + V0
    U = np.exp(-l1 * t) * W0 * l2 / (l1 - l2) - C * np.exp(-l2 * t) + V0 + W0 + U0
    A = - W0 * np.exp(-l1 * t) + W0 + A0
    return np.array([U, A])

def Fc(t, p, X0):
    l1 = p[0]
    l2 = p[1]
    W0, V0, U0, A0 = X0
    C = (l1 * W0 / (l1 - l2)) + V0
    W = W0 * np.exp(-l1 * t)
    V = -l1 * W0 * np.exp(-l1 * t) / (l1 - l2) + C * np.exp(-l2 * t)
    U = np.exp(-l1 * t) * W0 * l2 / (l1 - l2) - C * np.exp(-l2 * t) + V0 + W0 + U0
    A = - W0 * np.exp(-l1 * t) + W0 + A0
    return np.array([W, V, U, A])

def LogLikelihood(p):
    mu_U, mu_A = F(p)
    log_likelihood_U = np.sum(ss.poisson.logpmf(U_data, mu=mu_U))
    log_likelihood_A = np.sum(ss.poisson.logpmf(A_data, mu=mu_A))
    log_likelihood = log_likelihood_U + log_likelihood_A
    return log_likelihood

def LogPrior(p):
    p_l1 = gamma.logpdf(p[0], alp_l1, scale=1 / bet_l1)
    p_l2 = gamma.logpdf(p[1], alp_l2, scale=1 / bet_l2)
    return p_l1 + p_l2

def Energy(p):
    """ - logarithm of the posterior distribution (could it be proportional) """
    return -(LogLikelihood(p) + LogPrior(p))

def Supp(p):
    """ Check if theta is in the support of the posterior distribution"""
    rt = all(p>0)
    return rt

def SimInit():
    """ Function to simulate initial values for the gamma distribution """
    p_l1 = gamma.rvs(alp_l1, scale=1 / bet_l1)
    p_l2 = gamma.rvs(alp_l2, scale=1 / bet_l2)
    return np.array([p_l1, p_l2])

def Run_twalk(T):
    d = 2
    np.random.seed()
    start = int(.05 * T)  # Burning
    twalk = pytwalk(n=d, U=Energy, Supp=Supp)     # Open the t-walk object
    twalk.Run(T=T, x0=SimInit(), xp0=SimInit())   # Run the t-walk with two initial values for theta
    hpd_index = np.argsort(twalk.Output[:, -1])   # Maximum a posteriori index
    MAP = twalk.Output[hpd_index[0], :-1]         # MAP
    Out_s = twalk.Output[start:, :-1]
    mean_post = Out_s[start:, :].mean(axis=0)
    quantiles = np.quantile(Out_s[start:, :], axis=0, q=[0.05,0.5,0.95])
    energy = twalk.Output[:, -1]
    return Out_s, energy, MAP, mean_post, quantiles


def plot_post(index, Out_s , ax):
    fontsize=35
    Out_r = Out_s[:, index]  # Output without burning
    rt = ax.hist(Out_r, bins=15, density=True, label='Posterior')
    x = np.linspace(rt[1].min(), rt[1].max(), 10)

    if index == 0:
        title = '$\lambda_{v_1}$'
        ax.plot(x, gamma.pdf(x, a=alp_l1, scale=1 / bet_l1), 'r-', lw=5, alpha=0.6, label='Prior')
        ax.axvline(x=mean_post[0],ymin=0,ymax=1, color='k', label='Mean posterior')
        plt.xticks([0.005976, 0.005982, 0.005988])
        plt.yticks([20000,  80000, 140000])

    else:
        title = '$\lambda_{v_2}$'
        ax.plot(x, gamma.pdf(x, a=alp_l2, scale=1 / bet_l2), 'r-', lw=5, alpha=0.6, label='Prior')
        ax.axvline(x=mean_post[1], ymin=0, ymax=1, color='k', label='Mean posterior')
        plt.xticks([0.03258, 0.03261, 0.03264])
        plt.yticks([5000, 20000, 35000])

    ax.legend(frameon=False,fontsize=fontsize-5,loc=2)
    ax.set_xlabel(title, fontsize=fontsize+5)
    ax.tick_params(which='major', axis='x', labelsize=fontsize)
    ax.tick_params(which='major', axis='y', labelsize=fontsize)
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.set_ylabel('Density',fontsize=fontsize)

def plot_energy(ax, energy,burning):
    fontsize = 35
    if burning:
        plt.xticks([0, 4000, 8000])
        plt.yticks([44325, 44327, 44329])
    else:
        plt.xticks([0, 5000, 10000])
    ax.plot(energy,lw=2)
    ax.set_ylabel('Energy', fontsize=fontsize)
    ax.set_xlabel('Iterations', fontsize=fontsize)
    ax.tick_params(which='major', axis='x', labelsize=fontsize)
    ax.tick_params(which='major', axis='y', labelsize=fontsize)


def plot_Fig_S2(T, Out_s, workdir, out_fnam, energy):
    #plt.p
    start = int(.1 * T)
    fig, ax = plt.subplots(num=1, figsize=(14, 8))
    plot_post(index=0, Out_s=Out_s , ax=ax)
    fig.tight_layout()
    fig.savefig("%s%s_post_l1.png" % (workdir, out_fnam))

    fig, ax = plt.subplots(num=2, figsize=(14, 8))
    plot_post(index=1, Out_s=Out_s , ax=ax)
    fig.tight_layout()
    fig.savefig("%s%s_post_l2.png" % (workdir, out_fnam))

    fig, ax = plt.subplots(num=3, figsize=(14, 8))
    plot_energy(ax, energy, burning=False)
    fig.tight_layout()
    fig.savefig("%s%s_energy.png" % (workdir, out_fnam))

    fig, ax = plt.subplots(num=4, figsize=(14, 8))
    plot_energy(ax, energy[start:],burning=True)
    fig.tight_layout()
    fig.savefig("%s%s_energy_burning.png" % (workdir, out_fnam))

#--- Uncomment theses lines to run the MCMC
'''
T=10000
Out_s, energy, MAP, mean_post, quantiles = Run_twalk(T=T)
Output_MCMC = {"output": Out_s, "mean_post": mean_post, "quantiles": quantiles, "MAP": MAP,'energy': energy} # guardar la media posterior
Output_MCMC_file = open("Output_MCMC.pkl", "wb")
pickle.dump(Output_MCMC, Output_MCMC_file)
Output_MCMC_file.close()
'''

#--- Uncomment theses lines to plot Figure S2
'''
T=10000
Output_MCMC_file = open("Output_MCMC_1.pkl", "rb")
Output_MCMC = pickle.load(Output_MCMC_file)
Out_s = Output_MCMC["output"]
energy = Output_MCMC["energy"]
mean_post=Output_MCMC["mean_post"]
workdir = "./../"
clave = 'CA'
out_fnam = clave
plot_Fig_S2(T, Out_s, workdir, out_fnam, energy)

'''



