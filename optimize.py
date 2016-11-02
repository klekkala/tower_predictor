#!/usr/bin/python
#-*- coding: latin-1 -*-
"""This module contains pure Python implementations of the Levenberg-Marquardt algorithm for data fitting.
"""

import numpy
import matplotlib.pyplot as plt
from numpy import inner, diag, eye, Inf, dot
from numpy.linalg import norm, solve
import time, helper

def cost(guess, tower, elev, pop, land):

    x, y = guess[0:1]
    towerx, towery = tower[0:1]
    elevx, elevy = elev[0:1]
    popx, popy = pop[0:1]
    tower_distance = point_distance(x, y, towerx, towery)
    pop_distance = point_distance(x, y, popx, popy)
    elev_interp = point_interpolate(x, y, elevx, elevy)
    land_value = point_polygon(x, y, land)
    cost_value = numpy.concatenate(tower_distance, pop_distance, elev_interp, land_value)

    return cost_value



def fJ(guess, tower, elev, pop, land, b):
    "Calculation of function and Jacobian for one-dimensional Gaussian."
    x, y = guess[0:1]
    towerx, towery = tower[0:1]
    ##b=y
    f = cost(guess, tower, elev, pop, land)
    if 1:
        J = numpy.empty(shape = (2,)+x.shape, dtype = numpy.float_)
        J[0] = (towerx - x)/point_distance(x, y, towerx, towery)
        J[1] = (towery - y)/point_distance(x, y, towerx, towery)
        return f - b, J
    return f - b



def LMqr(fun, guess, tower, pop, elev, land, b,
         tau = 1e-3, eps1 = 1e-8, eps2 = 1e-8, kmax = 100,
         verbose = False):

    from scipy.linalg import lstsq
    import scipy.linalg

    """Implementation of the Levenberg-Marquardt algorithm in pure
    Python. Instead of using the normal equations this version uses QR
    factorization for enhanced accuracy. Significantly slower (factor
    2)."""
    f, J = fun(guess, tower, elev, pop, land, b)

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

       
        q, r = scipy.linalg.qr(Des, mode = 'economic')
        d4   = scipy.linalg.cho_solve( (r, False), -inner(J, f))
        print 'd4', d4, time.time() - tic


        print "norms"
        print norm(d4 - d)


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
            mu = mu*max(1.0/3, 1 - (2*rho - 1)**3)
            nu = 2
        else:
            mu = mu * nu
            nu = 2*nu

    else:
        reason = "max iter reached"

    if verbose:
        print reason
    return p



def minimize(guess, tower, pop, elev, land):

    b = feature_properties()

    return LMqr(fJ, guess, tower, pop, elev, land, b, verbose = True)


if __name__ == '__main__':
    print testfun()