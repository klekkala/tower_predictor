#!/usr/bin/python
#-*- coding: latin-1 -*-
"""This module contains pure Python implementations of the
Levenberg-Marquardt algorithm for data fitting.
"""

import numpy
import matplotlib.pyplot as plt
from numpy import inner, diag, eye, Inf, dot
from numpy.linalg import norm, solve


import time

def gauss1d(pars, x, v0 = 0):
    """calculate 1d gaussian.
    @return: difference of 1d gaussian and reference (data) values
    @param pars: parameters of gaussian. see source.
    @param x: x values
    @param v0: reference value
    """
    A, m, s, offs = pars[0:4]
    v = A*numpy.exp(- (x-m)**2 / (2*s**2)) + offs
    return v-v0

def Dgauss1d(pars, x, v=0):
    """
    calculated Jacobian matrix for 1d gauss
    """
    A, m, s, offs = pars[0:4]
    f = A*numpy.exp( - (x-m)**2 / (2*s**2))
    J = numpy.empty(shape = (4,)+x.shape, dtype = numpy.float_)
    J[0] = 1.0/A * f
    J[1] = f*(x-m)/s**2
    J[2] = f*(x-m)**2/s**3
    J[3] = 1
    return J

def fJ(pars, x, y = 0):
    "Calculation of function and Jacobian for one-dimensional Gaussian."
    A, m, s, offs = pars[0:4]
    f = A*numpy.exp( - (x-m)**2 / (2*s**2))
    if 1:
        J = numpy.empty(shape = (4,)+x.shape, dtype = numpy.float_)
        J[0] = 1.0/A * f
        J[1] = f*(x-m)/s**2
        J[2] = f*(x-m)**2/s**3
        J[3] = 1
        return f + (offs - y), J
    return f + (offs - y)


def rF(pars, x):
    """calculate all f_i and df_i/dp_j"""
    m, s = pars

    #F: function in parts which are then linearly combined to yield total function
    F = numpy.empty(shape = (2,) + x.shape)
    F[0] = numpy.exp( - (x-m)**2 / (2*s**2))
    F[1] = 1

    Fd = numpy.empty(shape = (2,) + F.shape)
    ##Ableitungen nach nichtlinearen Parametern
    #Fd[0]: Ableitungen der F[i] nach m
    Fd[0][0] = F[0] * (x-m)/(s**2)
    Fd[0][1] = 0

    Fd[1][0] = F[0] * (x-m)**2/(s**3)
    Fd[1][1] = 0

    return F, Fd

def fJr(pars, x, y = 0, calcJ = True):
    """
    calculate f and J for reduced system (only nonlinear parameters)
    """

    F, Fd = rF(pars, x)

    #calculate linear Parameters
    FtF = inner(F, F)
    Fty = inner(F, y)
    c = solve(FtF, Fty)

    #calculate residuum
    r = dot(c, F) - y

    if not calcJ:
        return r, c, F

    ##calculate complete Jacobian
    cd = numpy.empty(shape = (len(pars),) + c.shape)
    Jr = numpy.empty(shape = (len(pars),) + x.shape)
    for j in range(len(pars)):
        cd[j] = solve(FtF, inner(Fd[j], r) - inner(F, dot(c, Fd[j])))
        Jr[j] = dot(c, Fd[j]) + dot(cd[j], F)

    return r, Jr



def LMqr(fun, pars, args,
         tau = 1e-3, eps1 = 1e-8, eps2 = 1e-8, kmax = 100,
         verbose = False):

    from scipy.linalg import lstsq
    import scipy.linalg

    """Implementation of the Levenberg-Marquardt algorithm in pure
    Python. Instead of using the normal equations this version uses QR
    factorization for enhanced accuracy. Significantly slower (factor
    2)."""
    p = pars
    f, J = fun(p, *args)

    A = inner(J,J)
    g = inner(J,f)

    I = eye(len(p))

    k = 0; nu = 2
    mu = tau * numpy.max(diag(A))
    stop = norm(g, Inf) < eps1

    while not stop and k < kmax:
        k += 1

        if verbose:
            print "step %d: |f|: %9.3g mu: %g"%(k, norm(f), mu)

        tic = time.time()
        A = inner(J, J)
        g = inner(J, f)

        d = solve( A + mu*I, -g)
        print 'XX', d, time.time() - tic

        
        des = numpy.hstack((-f, numpy.zeros((len(p),))))
        Des = numpy.vstack((numpy.transpose(J),
                            numpy.sqrt(mu)*I))

        tic = time.time()
        d0, resids, rank, s = lstsq(Des, des)
        print 'd0', d0, time.time() - tic

        
        tic = time.time()
        #q, r = scipy.linalg.qr(Des, econ = True, mode = 'qr')
        #d4   = solve(r, inner(numpy.transpose(q), des))
        q, r = scipy.linalg.qr(Des, mode = 'economic')
        d4   = scipy.linalg.cho_solve( (r, False), -inner(J, f))
        print 'd4', d4, time.time() - tic

        
        

        tic = time.time()
        q, r = scipy.linalg.qr(numpy.transpose(J), mode = 'economic')
        d3 = solve( r + mu*numpy.linalg.inv(r.transpose()), -inner(numpy.transpose(q),f))
        #d3 = scipy.linalg.cho_solve( (r + mu*numpy.linalg.inv(r.transpose()), False),
        #                             -inner(numpy.transpose(q),f))
        print 'd3', d3, time.time() - tic

        print d - d0
        print d3 - d0
        print d4 - d0


        if norm(d) < eps2*(norm(p) + eps2):
            stop = True
            reason = 'small step'
            break

        pnew = p + d

        fnew, Jnew = fun(pnew, *args)
        rho = (norm(f) - norm(fnew))/inner(d, mu*d - g) # /2????

        if rho > 0:
            p = pnew
            #A = inner(Jnew, Jnew)
            #g = inner(Jnew, fnew)
            f = fnew
            J = Jnew
            if (norm(g, Inf) < eps1): # or norm(fnew) < eps3):
                stop = True
                reason = "small gradient"
                break
            print rho
            mu = mu*max(1.0/3, 1 - (2*rho - 1)**3)
            print "end"
            nu = 2
        else:
            mu = mu * nu
            nu = 2*nu

    else:
        reason = "max iter reached"

    if verbose:
        print reason
    return p


def testLMqr():
    pars = [1, 0.1, 1, 0.5]
    x = numpy.linspace(-5,5,1000001)
    y = gauss1d(pars, x) # + numpy.random.randn(len(x))
    pars2 = [1.1, 0.15, 1.3, 0.2]

    return LMqr(fJ, pars2, (x, y), verbose = True)



def cost(x):

	cost_value = 0
	for i in range(5):
		if x==point:
			sum_val = total_point_distance(x)

		elif x==line:
			sum_val = min_point_line(x)

		elif x==polygon:
			sum_val = val_point_polygon(x)

		cost_value = cost_value + (5-i)*sum_val
	return cost_value


def minimize(priority):
  x0 = np.array([0.0, 0.0])
  return res.x

if __name__ == '__main__':
    print testLMqr()