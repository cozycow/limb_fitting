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


def model(r, alpha, beta, epsilon, scale, bias, rsun, sigma, resolution=0.05, **kwargs):
    from scipy.signal import fftconvolve
    from scipy.ndimage import gaussian_filter

    rmax = int(np.ceil(np.max(r)))
    ri = np.arange(0, rmax + 1, resolution)
    qi = neckel(np.sqrt((1 - ri ** 2 / rsun ** 2).clip(0)))
    qi = gaussian_filter(qi, sigma / resolution)

    xi, yi = np.mgrid[-rmax:rmax+1, -rmax:rmax+1]
    r2 = xi ** 2 + yi ** 2
    Q = np.interp(np.sqrt(r2), ri, qi)
    P = 1 / (1 + r2 / alpha ** 2) ** beta
    P /= np.sum(P)
    Q_ = fftconvolve(Q, P, mode='same')
    Q_ = Q_ * epsilon + Q * (1 - epsilon)

    q = np.interp(r, np.arange(-rmax, rmax+1), Q_[rmax])
    return q * scale + bias


def scan(image, r0=0, h=200, phi0=0, phi1=360, **kwargs):
    from scipy.ndimage import map_coordinates

    if 'xc' not in kwargs and 'yc' not in kwargs and 'rsun' not in kwargs:
        xc, yc, rsun = find_center(image)
    else:
        xc = kwargs['xc']
        yc = kwargs['yc']
        rsun = kwargs['rsun']

    r, phi = np.mgrid[r0:rsun + h,phi0:phi1]
    Q = map_coordinates(image, (r * np.cos(phi * np.pi / 180) + xc, r * np.sin(phi * np.pi / 180) + yc),
                        order=3, mode='constant', cval=np.nan)
    profile = np.nanmedian(Q, axis=1)
    return np.arange(r0,rsun + h), profile


def fit_cld(image, **kwargs):
    from scipy.optimize import least_squares

    def residuals(args, r, profile, **kwargs):
        return np.nan_to_num(profile / model(r, *args, **kwargs) - 1)

    r, profile = scan(image, **kwargs)
    rsun = r[np.where(profile < 0.2)[0][0]]

    result = least_squares(residuals, np.array([1.5, 1.5, 0.25, 1, 0, rsun, 0.9]),
                           bounds=([0.1, 1, 0, 0.5, -0.1, rsun-5, 0.5], [3, 2, 1, 2, 0.1, rsun+5, 1.5]),
                           args=(r, profile), kwargs=kwargs)

    params = result.x
    fit = model(r, *params)
    return params, r, profile, fit