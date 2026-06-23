from limb_fitting import *
from interpolation import *


def realign(data, x0=None, y0=None, **kwargs):
    data_ = data.copy().reshape((-1, data.shape[-2], data.shape[-1]))

    if x0 is None and y0 is None:
        x0, y0, _ = find_center(data_[0], **kwargs)

    for i in range(len(data_)):
        xc, yc, _ = find_center(data_[i], **kwargs)
        dx, dy = x0 - xc, y0 - yc
        data_[i] = interp2d(data_[i], dx, dy, roll=True, **kwargs)

    return data_.reshape(data.shape)
