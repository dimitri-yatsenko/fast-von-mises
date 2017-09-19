"""
Fast two-peak von Mises fit
"""
__author__ = "Dimitri Yatsenko"

import numpy as np

nwidths = 64  # make a power of two
widths = np.logspace(0, 1, nwidths, base=30.0)


def g(c, w):
    """ von Mises peak """
    return np.exp(-w*(1-c))


def von_mises2(phi, a0, a1, a2, theta, w):
    """ two-peak von Mises """
    c = np.cos(phi-theta)
    return a0 + a1 * g(c, w) + a2 * g(-c,w)


def fit_von_mises2(phi, x):
    """
    :input phi: 1D vector of equidistally distributed angles between 0 and 2*pi
    :input x:  1D vector of response magnitudes at those angles
    :output: v, r2 - where v is the list of the fitted coefficients and r2 is squared error
    """
    # estimate theta with two-cosine fit
    s = x @ np.exp(2j*phi)
    theta = 0.5*np.angle(s)
    xm = x.mean()
    x = x - xm
    c = np.cos(phi-theta)

    def amps(width):
        # fit amplitudes
        G = np.stack((g(c, width), g(-c, width)))
        gm = G.mean(axis=1, keepdims=True)
        a = np.maximum(x @ np.linalg.pinv(G-gm), 0)
        d = x - a @ (G-gm)
        return d@d, a, gm, width, np.sign(d @ (a @ (G * np.stack((1-c, 1+c)))))

    # binary search for optimal width
    best = None
    bounds = [0, nwidths]
    while bounds[1]-bounds[0] > 1:
        mid = (bounds[0]+bounds[1])//2
        candidate = amps(widths[mid])
        if best is None or best[0]>candidate[0]:
            best = candidate
        bounds[1 if candidate[4]>0 else 0] = mid

    r2, a, gm, w, _ = best

    if a[0]<a[1]:
        a = a[[1,0]]
        theta = theta + np.pi

    return (xm-a@gm, a[0], a[1], theta % (2*np.pi), w), r2


def bootstrap_von_mises2(phi, x, shuffles=5000):
    v, r2 = fit_von_mises2(phi, x)
    return v, r2, np.array([fit_von_mises2(phi, np.random.choice(x, x.shape))[1] < r2 
                            for shuffle in range(shuffles)]).mean() + 0.5/shuffles
    