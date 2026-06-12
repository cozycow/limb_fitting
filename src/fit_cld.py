import numpy as np
from limb_fitting import find_center


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


def model(r, beta, epsilon, scale, bias, rsun, sigma, alpha=2, resolution=0.01, window=255):
    from scipy.signal import fftconvolve
    from scipy.ndimage import gaussian_filter

    rmax = int(np.ceil(np.max(r)))
    ri = np.arange(-rmax, rmax + 1, resolution)
    qi = neckel(np.sqrt((1 - ri ** 2 / rsun ** 2).clip(0)))
    qi = gaussian_filter(qi, sigma / resolution)

    xi, yi = np.mgrid[-rmax:rmax+1, -rmax:rmax+1]
    Q = np.interp(np.sqrt(xi ** 2 + yi ** 2), ri, qi)

    xi, yi = np.mgrid[-window:window+1,-window:window+1]
    r2 = xi ** 2 + yi ** 2
    P = 1 / (1 + r2 / alpha ** 2) ** beta
    P /= np.sum(P)
    Q_ = fftconvolve(Q, P, mode='same')
    Q_ = Q_ * epsilon + Q * (1 - epsilon)

    q = np.interp(r, np.arange(-rmax, rmax+1), Q_[rmax])
    return q * scale + bias


def scan(image, h=100):
    nx, ny = image.shape
    xc, yc, rsun = find_center(image)
    xi, yi = np.mgrid[:nx, :ny]
    ri = np.sqrt((xi - xc) ** 2 + (yi - yc) ** 2)
    r = np.arange(-0.5, np.floor(rsun + h), 1)

    profile = []
    for a, b in zip(r[:-1], r[1:]):
        t = np.where(np.all([ri > a, ri < b], axis=0))
        profile += [np.nanmedian(image[t])]

    profile = np.array(profile)
    profile /= np.nanpercentile(profile, 99)
    r = (r[:-1] + r[1:]) / 2
    return r, profile


def fit_cld(image, **kwargs):
    from scipy.optimize import least_squares

    def residuals(args, r, profile):
        return np.nan_to_num(profile / model(r, *args) - 1)

    xc, yc, rsun = find_center(image)
    r, profile = scan(image)
    result = least_squares(residuals, np.array([1.5, 0.25, 1, 0, rsun, 0.9]),
                           bounds=([1, 0, 0.5, -0.1, rsun-2, 0.5], [2, 1, 2, 0.1, rsun+2, 1.5]),
                           args=(r, profile), **kwargs)

    params = result.x
    fit = model(r, *params)
    return params, r, profile, fit