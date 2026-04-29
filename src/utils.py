import numpy as np
from interpolation import *


def bilinear(image, x, y):
    nx, ny = image.shape

    x_ = np.nan_to_num(np.floor(x), nan=nx).astype(np.int16)
    y_ = np.nan_to_num(np.floor(y), nan=ny).astype(np.int16)
    dx, dy = x - x_, y - y_

    image_ = np.zeros_like(x).astype(np.float32)
    for i in [0, 1]:
        for j in [0, 1]:
            q = np.abs((1 - i - dx) * (1 - j - dy))
            xi, yj = x_ + i, y_ + j
            temp = image[xi % nx, yj % ny] * q
            image_ += temp

    return image_


def crop(data, header):
    nx, ny = header['NAXIS2'], header['NAXIS1']
    x0, y0 = header['PXBEG2'] - 1, header['PXBEG1'] - 1
    return data[...,x0:x0+nx, y0:y0+ny]


def crop_grid(xi, yi, header):
    nx, ny = header['NAXIS2'], header['NAXIS1']
    x0, y0 = header['PXBEG2'] - 1, header['PXBEG1'] - 1
    return xi[x0:x0 + nx, y0:y0 + ny] - x0, yi[x0:x0 + nx, y0:y0 + ny] - y0


def undistort(image, header, xd, yd, **kwargs):
    from interpolation import interp2d
    xd_, yd_ = crop_grid(xd, yd, header)
    return interp2d(image, xd_, yd_, **kwargs)


def kll(data, shifts, niter=20, **kwargs):
    F = np.zeros_like(data[0])
    C0 = shifts[0]

    for iter in range(niter):
        D = np.zeros_like(F)
        M = np.zeros_like(F)
        for Di, Ci in zip(data, shifts):
            Di_ = interp2d(Di - F, *(C0 - Ci), roll=True, **kwargs)
            M += ~np.isnan(Di_)
            D += np.nan_to_num(Di_ - D) / M.clip(1)

        D[M == 0] = np.nan

        F_ = np.zeros_like(F)
        W = np.zeros_like(F)

        for Di, Ci in zip(data, shifts):
            Di_ = Di - interp2d(D, *(Ci - C0), roll=True, **kwargs)

            Wi_ = ~np.isnan(Di_)
            W += np.nan_to_num(Wi_)
            with np.errstate(invalid='ignore'):
                F_ += np.nan_to_num((Di_ - F_) * Wi_ / W)

        F_[W == 0] = np.nan
        F = F_.copy()

    return F
