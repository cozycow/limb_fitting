import numpy as np



def neckel(mu):
    p = [0.48767921486914473,
         -1.6848471461910317,
         2.355950448408068,
         -1.827014432405401,
         1.3540877312885482,
         0.31414418403067174]
    return np.polyval(p, mu) * (mu > 0)


def moffat(x, alpha=1., beta=1.):
    return (1 + (x / alpha) ** 2) ** (-beta)


def model(r, sigma, alpha=1, beta=1, epsilon=1., scale=1., bias=0., rsun=1.):
    from scipy.signal import fftconvolve

    dr = 0.05
    rmax = np.ceil(rsun + 1000)
    R = np.arange(-rmax, rmax + dr / 2, dr)
    Q = neckel(np.sqrt((1 - (R / rsun) ** 2).clip(0)))
    G = np.exp(-R ** 2 / 2 / sigma ** 2)
    G /= np.sum(G)
    Q = fftconvolve(Q, G, mode='same')
    P = moffat(R, alpha=alpha, beta=beta)
    P /= np.sum(P)
    p = fftconvolve(Q, P, mode='same')
    q = p * epsilon + (1 - epsilon) * Q
    q = np.interp(r, R, q)
    return q * scale + bias


def fit_cld(image):
    from scipy.optimize import curve_fit
    from limb_fitting import find_center

    nx, ny = image.shape
    xc, yc, rsun = find_center(image)
    xi, yi = np.mgrid[:nx, :ny]

    ri = np.sqrt((xi - xc) ** 2 + (yi - yc) ** 2)
    r = np.arange(-0.5, np.floor(rsun + 50), 1)

    q = []
    for a, b in zip(r[:-1], r[1:]):
        t = np.where(np.all([ri > a, ri < b], axis=0))
        q += [np.nanmedian(image[t])]

    q = np.array(q)
    q /= np.nanpercentile(q, 99)
    r = (r[:-1] + r[1:]) / 2

    params, _ = curve_fit(model, r, q, bounds=([0.7, 1, 0.6, 0.1, 0.8, 0, rsun - 2], [1.2, 5, 1.2, 0.9, 1.2, 0.2, rsun + 2]),
                          nan_policy='omit', sigma=np.sqrt(q))
    return r, q, params