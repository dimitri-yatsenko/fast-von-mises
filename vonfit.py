"""
Fast two-peak von Mises fit
"""
__author__ = "Dimitri Yatsenko"

import numpy as np

# von Mises peak
def g(c, w):
    return np.exp(-w*(1-c))


# two-peak von Mises
def von_mises2(phi, a0, a1, a2, theta, w):
    c = np.cos(phi-theta)
    return a0 + a1 * g(c, w) + a2 * g(-c,w)


def fit_von_mises2(phi, x):
    """
    :input phi: 1D vector of equidistally distributed angles between 0 and 2*pi
    :input x:  1D vector of response magnitudes at those angles
    :output: v, r2 - where w are the fitted coefficients and r2 is squared error
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
    bounds = [0, fit_von_mises2.nwidths]
    while bounds[1]-bounds[0] > 1:
        mid = (bounds[0]+bounds[1])//2
        candidate = amps(fit_von_mises2.widths[mid])
        if best is None or best[0]>candidate[0]:
            best = candidate
        bounds[1 if candidate[4]>0 else 0] = mid

    r2, a, gm, w, _ = best

    if a[0]<a[1]:
        a = a[[1,0]]
        theta = theta + np.pi

    return (xm-a@gm, a[0], a[1], theta % (2*np.pi), w), r2

fit_von_mises2.nwidths = 64  # make a power of two
fit_von_mises2.widths = np.logspace(0, 1, fit_von_mises2.nwidths, base=30.0)
