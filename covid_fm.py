import numpy as np


class fm:
    
    def __init__(self, m, num_state_vars):
        self.m = m
        self.num_state_vars = num_state_vars
        self.X0 = np.zeros(self.num_state_vars)  # Initial conditions:
        self.weigths = np.ones(11)               # quadrature weights
        self.weigths[0] = 0.5
        self.weigths[-1] = 0.5
        self.dt = 1.0 / (10.0)                   # size step for quadrature
        self.SetTime()                           # Set the time ranges for quadrature and the fm

    def SetTime(self, shift=0):
        self.time = np.linspace(-1.0, self.m - 1 + shift, num=(self.m + shift) + 1, endpoint=True)  # observation time
        self.nn = len(self.time)
        self.t_quad = np.linspace(-1.0, self.m - 1 + shift, num=10 * (self.m + shift) + 1, endpoint=True)  # Grid for quadrature
        self.n_quad = len(self.t_quad)
        self.result_I = np.zeros(self.nn-1)   # To hold result of quadrature
        self.result_D = np.zeros(self.nn-1)   # To hold result of quadrature
        self.result_H1 = np.zeros(self.nn-1)  # To hold result of quadrature

    def rhs(self, x, t, p):
        pass
    
    def solve_plain(self, p):
        pass
    
    def solve(self, p):
        pass


AuxMatrixLatexHead = r"""
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Latex file automatically created by covid_fm.AuxMatrix
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\documentclass[11pt]{article}

%% tickz stuff %%
\usepackage{tikz} 
\usetikzlibrary{fit,positioning}
\usetikzlibrary{matrix}
\usetikzlibrary{calc}
\usetikzlibrary{arrows}
\tikzstyle{int}=[draw, fill=white!8, minimum size=2em]
\tikzstyle{init} = [pin edge={to-,thin,black}]
\newcommand\encircle[1]{%
  \tikz[baseline=(X.base)]
    \node (X) [draw, shape=circle, inner sep=0.5] {\strut #1};}
    
%%%<
\usepackage{verbatim}
\usepackage[active,tightpage]{preview}
\PreviewEnvironment{tikzpicture}
\setlength\PreviewBorder{5pt}%
%%%>

\begin{document}
\pagestyle{empty}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\begin{tikzpicture}[node distance=2.5cm,auto,>=latex']

"""

AuxMatrixLatexTail = r"""
\end{tikzpicture}

\end{document}
"""

#import os

class AuxMatrix:
    def __init__(self, names=None, num_state_vars=None, prn=False, tex_fnam=None, pdflatex="/Library/TeX/texbin/pdflatex"):

        if names == None:
            self.names = [r"V^{%d}" % (i,) for i in range(num_state_vars)]
        else:
            self.names = names.split()
        self.names_org = self.names.copy()

        if num_state_vars == None:
            self.q = len(self.names)
        else:
            self.q = num_state_vars
        self.n = self.q
        # Length of Erlang list for each variable, we start with 1 (no Erlang list in fact)
        self.Erlang_len = [1] * self.n
        self.M = np.diagflat(-np.ones(self.q), k=0)  # Initial matrix, every var has an exit (-1 in the diagonal)
        self.par_mask = np.diagflat(np.ones(self.q), k=0)
        self.par_vec = np.zeros((self.q, 1))
        self.prn = prn

        # To produce the latex diagram of the graph
        if tex_fnam == None:
            self.tex_f = open('AuxMatrixGraph.tex', 'w')
            self.tex_fnam = 'AuxMatrixGraph.tex'
        else:
            self.tex_f = open(tex_fnam, 'w')
            self.tex_fnam = tex_fnam
        self.pdflatex = pdflatex

    def ConvertVar(self, v):
        if isinstance(v, int):
            return v
        else:
            return self.names_org.index(v)

    def PrintMatrix(self, j1=None, j2=None):

        if j1 == None:
            j1 = 0
        if j2 == None:
            j2 = self.q

        print("\n%9s" % (" ",), end=' ')
        for j in range(j1, j2):
            print("%9s" % (self.names[j],), end=' ')
        print("")
        for i in range(j1, j2):
            print("%9s" % (self.names[i],), end=' ')
            for j in range(j1, j2):
                print("%9.2f" % (self.M[i, j],), end=' ')
            print("")

    def BaseVar(self, v):
        v = self.ConvertVar(v)
        print(AuxMatrixLatexHead, file=self.tex_f)
        print(r"\node [int] (%s) {$%s$};" % (self.names[v], self.names[v]), file=self.tex_f)
        print("", file=self.tex_f)

    def SplitExit(self, v, v1, v2, prob, prob_symb=None):
        v = self.ConvertVar(v)
        v1 = self.ConvertVar(v1)
        v2 = self.ConvertVar(v2)
        self.M[v+1, v] = 0     # Delete its exit
        self.M[v1, v] = prob   # Split
        self.M[v2, v] = 1-prob

        if prob_symb == None:
            p = "%4.2f" % (prob,)
            p1 = "%4.2f" % (1-prob,)
        else:
            p = prob_symb[0]
            p1 = prob_symb[1]
        print(r"\node [int] (%s) [above right of = %s] {$%s$};" % (self.names[v1], self.names[v], self.names[v1]), file=self.tex_f)
        print(r"\node [int] (%s) [below right of = %s] {$%s$};" % (self.names[v2], self.names[v], self.names[v2]), file=self.tex_f)
        print(r"\path[->] (%s) edge node  {$%s$} (%s);" % (self.names[v], p, self.names[v1]), file=self.tex_f)
        print(r"\path[->] (%s) edge node  {$%s$} (%s);" % (self.names[v], p1, self.names[v2]), file=self.tex_f)
        print("", file=self.tex_f)

        if self.prn:
            print("%s --> split --> %s (%4.2f) and --> %s (%4.2f)" %\
                  (self.names[v], self.names[v1], prob, self.names[v2], 1-prob))

    def Exit(self, v, v1, w=1):
        v = self.ConvertVar(v)
        v1 = self.ConvertVar(v1)
        self.M[v1, v] = w
        print(r"\node [int] (%s) [right of = %s] {$%s$};" % (self.names[v1], self.names[v], self.names[v1]), file=self.tex_f)
        print(r"\path[->] (%s) edge node {} (%s);" % (self.names[v], self.names[v1]), file=self.tex_f)
        print("", file=self.tex_f)

        if self.prn:
            print("%s --> %s" % (self.names[v], self.names[v1]))

    def NoExit(self, v):
        """Variable v has no exit."""
        v = self.ConvertVar(v)
        self.M[v, v] = 0

    def End(self):
        """Does nothing to the matrix and flushes the latex output."""
        print(AuxMatrixLatexTail, file=self.tex_f)
        print("", file=self.tex_f)
        self.tex_f.close()

    def SplitErlang(self, v_list, m_list):
        if isinstance(m_list, int):
            m_list = [m_list] * len(v_list)  # Same m for all
        # Sort in ascending order the variables
        v_list = [self.ConvertVar(v) for v in v_list]
        tmp_L = [(v_list[i], i) for i in range(len(v_list))]
        tmp_L.sort()
        v_list, permutation = zip(*tmp_L)   # Sorted list of vars and their corresponding
        m_list = [m_list[permutation[i]] for i in range(len(m_list))]  # Erlang list lengths
        # Add the Erlang lists
        E_sum = 0
        for k, v_org in enumerate(v_list):  # v_org  original variable number
            m = m_list[k]
            if m == 1:
                continue  # Nothing to do
            elif m < 1:
                raise NameError('covid_fm: Negative Erlang list')

            v = v_org + E_sum  # k*(m-1)
            nm = self.names[v]
            l = 0
            self.names[v] += r"_{%d}" % (l,)
            self.par_mask[v, v_org] = m
            for i in range(v+1, v+m):
                self.M = np.insert(self.M, i, 0, axis=1)
                self.M = np.insert(self.M, i, 0, axis=0)
                self.q += 1
                self.M[i:,  i] = self.M[i:, i-1]  # Move the column below, to move all exists
                self.M[i:, i-1] *= 0  # Change all previous exists to zero
                self.M[i, i-1] = 1    # Add an entry to new variable
                self.M[i,   i] = -1   # Add its exit, to the previous exit of variable v
                l += 1
                self.names.insert(i, nm + r"_{%d}" % (l,))
                self.par_mask = np.insert(self.par_mask, i, 0, axis=0)
                self.par_mask[i, v_org] = m
            self.Erlang_len[v_org] = m
            E_sum += m-1  # Add m-1 state variables
        return True

    def GetE_range(self, v):
        v = self.ConvertVar(v)
        return self.Erlang_len[v]

    def SelectMask(self, v, E_range=[0], as_col_vec=False):
        v = self.ConvertVar(v)

        if isinstance( E_range, str):
            if E_range == 'all':
                E_range = range(self.Erlang_len[v])
            else:
                raise NameError('covid_fm: Unknown option E_range=%s' % (E_range,))
        if max(E_range) > self.Erlang_len[v]:
            raise NameError('covid_fm: var in Erlang list beyond list length %d:' % (self.Erlang_len[v],), E_range)

        # Make a vector with a 1 at row v in a column vector of zeros with self.n rows
        tmp = np.zeros((self.n, 1))
        tmp[v, 0] = 1
        # with the mask we obtain all variables in the Erlang list (possibly length 1)
        tmp2 = self.par_mask @ tmp
        # Where is the first variable in the list:
        i0 = np.where(tmp2 != 0)[0][0]
        # Slect the variables in the Erlang list
        tmp2 *= 0
        for i in E_range:
            tmp2[i0+i] = 1
        if as_col_vec:
            return tmp2
        else:
            return tmp2.flatten()


