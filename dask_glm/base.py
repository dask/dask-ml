from __future__ import absolute_import, division, print_function

import dask.array as da
import dask.dataframe as dd
from multipledispatch import dispatch
import numpy as np
import pandas as pd
from scipy.stats import chi2

def sigmoid(x):
    '''Sigmoid function of x.'''
    return 1/(1+da.exp(-x))

@dispatch(np.ndarray,np.ndarray)
def dot(A,B):
    return np.dot(A,B)

@dispatch(da.Array,da.Array)
def dot(A,B):
    return da.dot(A,B)

class Optimizer(object):

    def initialize(self, size, value=None, method=None):
        '''Method for setting the initialization.'''

        if value:
            self.init = value
        elif method=='random':
            self.init = np.random.normal(0,1,size)
        else:
            self.init = np.zeros(size)

        return self

    def hessian(self):
        raise NotImplementedError

    def gradient(self):
        raise NotImplementedError

    def func(self):
        raise NotImplementedError

    def bfgs(self, verbose=True, max_steps=100):
        recalcRate = 10
        stepSize = 1.0
        stepGrowth = 1.25
        beta = self.init
        M = beta.shape[0]
        Hk = np.eye(M)

        if verbose:
            print('##       -f        |df/f|    step')
            print('----------------------------------------------')

        for k in range(max_steps):

            if k % recalcRate==0:
                Xbeta = self.X.dot(beta)
                func = self.func(Xbeta)

            gradient = self.gradient(Xbeta)

            if k:
                yk += gradient
                rhok = 1/yk.dot(sk)
                adj = np.eye(M) - rhok*sk.dot(yk.T)
                Hk = adj.dot(Hk.dot(adj.T)) + rhok*sk.dot(sk.T)

            step = Hk.dot(gradient)
            steplen = step.dot(gradient)
            Xstep = self.X.dot(step)

            Xbeta, func, steplen, step, Xstep = da.compute(
                    Xbeta, func, steplen, step, Xstep)

            # Compute the step size
            if k==0:
                stepSize, beta, Xbeta, fnew = self._backtrack(func,
                    beta, Xbeta, step, Xstep,
                    stepSize, steplen, **{'backtrackMult' : 0.1,
                        'armijoMult' : 1e-4})
            else:
                stepSize, beta, Xbeta, fnew = self._backtrack(func,
                    beta, Xbeta, step, Xstep,
                    stepSize, steplen, **{'armijoMult' : 1e-4})

            yk = -gradient
            sk = -stepSize*step
            stepSize = 1.0
            df = func-fnew
            func = fnew

            if stepSize == 0:
                if verbose:
                    print('No more progress')

            df /= max(func, fnew)
            if verbose:
                print('%2d  %.6e %9.2e  %.1e' % (k + 1, func, df, stepSize))
            if df < 1e-14:
                print('Converged')
                break

        return beta

    def _check_convergence(self, old, new, tol=1e-4, method=None):
        coef_change = np.absolute(old - new)
        return not np.any(coef_change>tol)

    def fit(self, X, y, method=None, **kwargs):
        raise NotImplementedError

    def _newton_step(self,curr,Xcurr):

        hessian = self.hessian(Xcurr)
        grad = self.gradient(Xcurr)

        # should this be dask or numpy?
        step, *_ = da.linalg.lstsq(hessian, grad)
        beta = curr - step
        
        return beta.compute()

    def newton(self):
    
        beta = self.init
        Xbeta = self.X.dot(beta)

        iter_count = 0
        converged = False

        while not converged:
            beta_old = beta
            beta = self._newton_step(beta,Xbeta)
            Xbeta = self.X.dot(beta)
            iter_count += 1
            
            converged = (self._check_convergence(beta_old, beta) & (iter_count<self.max_iter))

        return beta

    def gradient_descent(self, max_steps=100, verbose=True):
        recalcRate = 10
        stepSize = 1.0
        stepGrowth = 1.25
        beta = self.init

        if verbose:
            print('##       -f        |df/f|    step')
            print('----------------------------------------------')

        for k in range(max_steps):

            if k % recalcRate==0:
                Xbeta = self.X.dot(beta)
                func = self.func(Xbeta)

            gradient = self.gradient(Xbeta)
            steplen = (gradient**2).sum()
            Xgradient = self.X.dot(gradient)

            Xbeta, func, gradient, steplen, Xgradient = da.compute(
                    Xbeta, func, gradient, steplen, Xgradient)

            # Compute the step size
            if k==0:
                stepSize, beta, Xbeta, fnew = self._backtrack(func,
                    beta, Xbeta, gradient, Xgradient,
                    stepSize, steplen, **{'backtrackMult' : 0.1})
            else:
                stepSize, beta, Xbeta, fnew = self._backtrack(func,
                    beta, Xbeta, gradient, Xgradient,
                    stepSize, steplen)

            stepSize *= stepGrowth
            df = func-fnew
            func = fnew

            if stepSize == 0:
                if verbose:
                    print('No more progress')

            df /= max(func, fnew)
            if verbose:
                print('%2d  %.6e %9.2e  %.1e' % (k + 1, func, df, stepSize))
            if df < 1e-14:
                print('Converged')
                break

        return beta

    ## this is currently specific to linear models
    def _backtrack(self, curr_val, curr, Xcurr, 
        step, Xstep, stepSize, steplen, **kwargs):

        ## theres got to be a better way...
        params = {'backtrackMult' : 0.5,
            'armijoMult' : 0.1}

        params.update(kwargs)
        backtrackMult = params['backtrackMult']
        armijoMult = params['armijoMult']
        Xbeta = Xcurr

        for ii in range(100):
            beta = curr - stepSize*step

            if ii and np.array_equal(curr, beta):
                stepSize = 0
                break
            Xbeta = Xcurr - stepSize*Xstep

            func = self.func(Xbeta)
            df = curr_val - func
            if df >= armijoMult * stepSize * steplen:
                break
            stepSize *= backtrackMult

        return stepSize, beta, Xbeta, func

    def __init__(self, max_iter=50, init_type='zeros'):
        self.max_iter = 50

class Model(Optimizer):
    '''Class for holding all output statistics.'''

    def fit(self,method='newton',**kwargs):
        methods = {'newton' : self.newton, 
            'gradient_descent' : self.gradient_descent,
            'BFGS' : self.bfgs}

        self.coefs = methods[method]()
        self._pvalues()

        return self

    def _pvalues(self, names={}):
        H = self.hessian(self.X.dot(self.coefs))
        covar = np.linalg.inv(H.compute())
        variance = np.diag(covar)
        self.se = variance**0.5
        self.chi = (self.coefs / self.se)**2
        chi2_cdf = np.vectorize(lambda t : 1-chi2.cdf(t,1))
        self.pvals = chi2_cdf(self.chi)

    def summary(self):
        if hasattr(self, 'names'):
            out = pd.DataFrame({'Coefficient' : self.coefs,
                'Std. Error' : self.se,
                'Chi-square' : self.chi,
                'p-value' : self.pvals}, index=self.names)
        else:
            out = pd.DataFrame({'Coefficient' : self.coefs,
                'Std. Error' : self.se,
                'Chi-square' : self.chi,
                'p-value' : self.pvals})
        return out

    def __init__(self, X, y, **kwargs):
        self.max_iter = 50

        if isinstance(X, dd.DataFrame):
            self.names = X.columns
            self.X = X.values
            M = self.X.shape[1]
        elif isinstance(X, dd.Series):
            self.names = [X.name]
            self.X = X.values[:, None]
            M = 1
        else:
            self.X = X
            M = self.X.shape[1]

        if isinstance(y, dd.DataFrame):
            self.y_name = y.columns[0]
            self.y = y.values[:,0]
        elif isinstance(y, dd.Series):
            self.y_name = y.name
            self.y = y.values
        else:
            self.y = y

        self = self.initialize(M)
