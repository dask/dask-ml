from __future__ import absolute_import, division, print_function

from dask_glm.utils import *
import dask.array as da
import dask.dataframe as dd
from multipledispatch import dispatch
import numpy as np
import pandas as pd
from scipy.stats import chi2

class Optimizer(object):
    '''Optimizer class for fitting linear models.

    Optional initialization arguments:

    max_iter (int): Maximum number of iterations allowed. Default is 50.
    '''

    def initialize(self, size, value=None, method=None):
        '''Set the initialization ('init' attribute).

        Required arguments:
        size (integer) : determines the length / size of the initialization vector
        
        Keyword arguments:
        value (array) : sets the initialization to the given value
        method (string) : determines alternate initialization routines; currently only
            supports 'random'
        '''

        if value:
            self.init = value
        elif method=='random':
            self.init = np.random.normal(0,1,size)
        else:
            self.init = np.zeros(size)

        return self

    def prox(self):
        raise NotImplementedError

    def hessian(self):
        raise NotImplementedError

    def gradient(self):
        raise NotImplementedError

    def func(self):
        raise NotImplementedError

    def bfgs(self, X, y):
        recalcRate = 10
        stepSize = 1.0
        stepGrowth = 1.25
        beta = self.init
        M = beta.shape[0]
        Hk = np.eye(M)

        for k in range(self.max_iter):

            if k % recalcRate==0:
                Xbeta = X.dot(beta)
                func = self.func(Xbeta, y)

            gradient = self.gradient(Xbeta, y)

            if k:
                yk += gradient
                rhok = 1/yk.dot(sk)
                adj = np.eye(M) - rhok*sk.dot(yk.T)
                Hk = adj.dot(Hk.dot(adj.T)) + rhok*sk.dot(sk.T)

            step = Hk.dot(gradient)
            steplen = step.dot(gradient)
            Xstep = X.dot(step)

            Xbeta, func, steplen, step, Xstep, y0 = da.compute(
                    Xbeta, func, steplen, step, Xstep, y)

            # Compute the step size
            if k==0:
                stepSize, beta, Xbeta, fnew = self._backtrack(func,
                    beta, Xbeta, step, Xstep,
                    stepSize, steplen, y0, **{'backtrackMult' : 0.1,
                        'armijoMult' : 1e-4})
            else:
                stepSize, beta, Xbeta, fnew = self._backtrack(func,
                    beta, Xbeta, step, Xstep,
                    stepSize, steplen, y0, **{'armijoMult' : 1e-4})

            yk = -gradient
            sk = -stepSize*step
            stepSize = 1.0
            df = func-fnew
            func = fnew

            if stepSize == 0:
                if verbose:
                    print('No more progress')

            df /= max(func, fnew)
            if df < 1e-8:
                print('Converged')
                break

        return beta

    def _check_convergence(self, old, new, tol=1e-4, method=None):
        coef_change = np.absolute(old - new)
        return not np.any(coef_change>tol)

    def fit(self, X, y, method=None, **kwargs):
        raise NotImplementedError

    def _newton_step(self,curr,Xcurr, y):

        hessian = self.hessian(Xcurr, y)
        grad = self.gradient(Xcurr, y)

        # should this be dask or numpy?
        # currently uses Python 3 specific syntax
        step, *_ = da.linalg.lstsq(hessian, grad)
        beta = curr - step
        
        return beta.compute()

    def newton(self, X, y):
    
        beta = self.init
        Xbeta = X.dot(beta)

        iter_count = 0
        converged = False

        while not converged:
            beta_old = beta
            beta = self._newton_step(beta,Xbeta,y)
            Xbeta = X.dot(beta)
            iter_count += 1
            
            converged = (self._check_convergence(beta_old, beta) & (iter_count<self.max_iter))

        return beta

    def gradient_descent(self, X, y, tol=1e-8, line_search='backtrack',**kwargs):
        '''Standard Gradient Descent method.'''

        recalcRate = 10
        stepSize = 1.0
        beta = self.init
    
        y = y.compute()

        for k in range(self.max_iter):

            if k % recalcRate==0:
                Xbeta = X.dot(beta)
                func = self.func(Xbeta, y)

            gradient = self.gradient(Xbeta, y)
            steplen = (gradient**2).sum()
            Xgradient = X.dot(gradient)

            Xbeta, func, gradient, steplen, Xgradient = da.compute(
                    Xbeta, func, gradient, steplen, Xgradient)


            if k:
                stepSize *= osteplen / steplen

            beta_old = beta

            # Compute the step size
            if line_search=='backtrack':
                if k==0:
                    stepSize, beta, Xbeta, fnew = self._backtrack(func,
                        beta, Xbeta, gradient, Xgradient,
                        stepSize, steplen, y, **{'backtrackMult' : 0.1})
                else:
                    stepSize, beta, Xbeta, fnew = self._backtrack(func,
                        beta, Xbeta, gradient, Xgradient,
                        stepSize, steplen, y)
            else:
                stepSize = kwargs['alpha']
                fnew = self.func(Xbeta - stepSize*Xgradient, y)

            osteplen = steplen
#            stepSize *= stepGrowth
            df = func-fnew
            func = fnew

            if stepSize == 0:
                if verbose:
                    print('No more progress')

            ## is this the right convergence criterion?
            df /= max(func, fnew)
            if df < tol:
                print('Converged')
                break

        return beta

    ## this is currently specific to linear models
    def _backtrack(self, curr_val, curr, Xcurr, 
        step, Xstep, stepSize, steplen, y, **kwargs):

        ## theres got to be a better way...
        params = {'backtrackMult' : 0.1,
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

            func = self.func(Xbeta, y)
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

        self.coefs = methods[method](self.X, self.y)
        self._pvalues()

        return self

    def _pvalues(self, names={}):
        H = self.hessian(self.X.dot(self.coefs), self.y)
        covar = np.linalg.inv(H.compute())
        variance = np.diag(covar)
        self.se = variance**0.5
        self.chi = (self.coefs / self.se)**2
        chi2_cdf = np.vectorize(lambda t : 1-chi2.cdf(t,1))
        self.pvals = chi2_cdf(self.chi)

    def summary(self):
        if hasattr(self, 'names'):
            out = pd.DataFrame({'coefficient' : self.coefs,
                'std_error' : self.se,
                'chi_square' : self.chi,
                'p_value' : self.pvals}, index=self.names)
        else:
            out = pd.DataFrame({'coefficient' : self.coefs,
                'std_error' : self.se,
                'chi_square' : self.chi,
                'p_value' : self.pvals})
        return out[['coefficient', 'std_error', 'chi_square', 'p_value']]

    def __init__(self, X, y, reg=None, **kwargs):
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

            ##FIXME
#            self.y._chunks = ((self.y.compute().shape[0],),)
        elif isinstance(y, dd.Series):
            self.y_name = y.name
            self.y = y.values

            ##FIXME
#            self.y._chunks = ((self.y.compute().shape[0],),)
        else:
            self.y = y

        self = self.initialize(M)

class Prior(object):

    def gradient(self, beta):
        raise NotImplementedError

    def hessian(self, beta):
        raise NotImplementedError

    def func(self, beta):
        raise NotImplementedError

    def prox(self, beta):
        raise NotImplementedError

    def __init__(self):
        
        return self

class RegularizedModel(Model):

    def gradient(self, Xbeta, beta):
        return self.base.gradient(Xbeta) + self.prior.gradient(beta)

    def hessian(self, Xbeta, beta):
        return self.base.hessian(Xbeta) + self.prior.hessian(beta)

    def func(self, Xbeta, beta):
        return self.base.func(Xbeta) + self.prior.func(beta)

    def __init__(self, base_model, prior, **kwargs):
        self.base = base_model
        self.prior = prior
        self.X = base_model.X
        self.y = base_model.y
