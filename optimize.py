#!/usr/bin/python
#-*- coding: latin-1 -*-
"""This module contains pure Python implementations of the Levenberg-Marquardt algorithm for data fitting.
"""

import numpy
import matplotlib.pyplot as plt
from numpy import inner, diag, eye, Inf, dot
from numpy.linalg import norm, solve
import time, helper

def cost(guess, towerx, towery, elevx, elevy, popx, popy, landx, landy):

    x, y = guess[0:1]

    tower_distance = point_distance(x, y, towerx, towery)
    pop_distance = point_distance(x, y, popx, popy)
    elev_interp = point_interpolate(x, y, elevx, elevy)
    land_value = point_polygon(x, y, land)
    cost_value = numpy.concatenate(tower_distance, pop_distance, elev_interp, land_value)

    return cost_value



def fJ(guess, tower, elev, pop, land, b):
    "Calculation of function and Jacobian for one-dimensional Gaussian."
    x, y = guess[0:1]

    f = cost(guess, tower, elev, pop, land)
    if 1:
        J = numpy.empty(shape = (2,)+x.shape, dtype = numpy.float_)
        tower_jacobx = (towerx - x)/point_distance(x, y, towerx, towery)
        tower_jacoby = (towery - y)/point_distance(x, y, towerx, towery)

        pop_jacobx = (popx - x)/point_distance(x, y, towerx, towery)
        pop_jacoby = (popy - y)/point_distance(x, y, towerx, towery)
        
        elev_jacobx = elev_jacoby = numpy.zeros(1)
        cost_jacoby = cost_jacoby = numpy.zeros(1)
        J[0] = numpy.concatenate(tower_jacobx, pop_jacobx, elev_jacobx, cost_jacobx)
        J[1] = numpy.concatenate(tower_jacoby, pop_jacoby, elev_jacoby, cost_jacoby)

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



def minimize(guess, cell_feat, cell_attr, pop_feat, pop_attr, elev_feat, elev_attr, land_feat, land_attr):

    b = feature_properties(cell_attr, pop_attr, elev_attr, land_attr)

    return LMqr(fJ, guess, tower_feat, pop_feat, elev_feat, land_feat, b, verbose = True)

def testfun():
    guess = [500,500]
    tower = [[-2579.06, 1554.85], [-2426.5, 1476.84], [-2532.25, 1390.16], [-2438.64, 1377.16]]
    pop = [[-2439.5, 1483.13], [-2364.09, 1471.86], [-2338.52, 1377.37], [-2248.81, 1429.38]]
    elev = [[-2541.79, 1506.75], [-2476.78, 1446.94], [-2340.69, 1551.82], [-2366.69, 1383.66], [-2310.35, 1481.61]]

    cost = [[[-2614.6, 1589.53], [-2424.77, 1612.06], [-2273.94, 1453.44], [-2347.62, 1356.35], [-2599, 1332.95], [-2614.6, 1589.53]], [[-2272.21, 1455.17], [-2177.73, 1399.69], [-2345.02, 1358.09], [-2272.21, 1455.17]], [[-2423.9, 1612.93], [-2234.07, 1591.26], [-2275.68, 1456.9], [-2423.9, 1612.93]]]
    cell_attr = [23, 34, 55, 67]
    pop_attr elev_attr, land_attr
    return LMqr(fJ, guess, tower, pop, elev, land, b, verbose = True)


if __name__ == '__main__':
    print testfun()
