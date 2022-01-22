import numpy as np
import pywt  # PyWavelets-1.1.1
import torch

wavelet = "haar"
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def dwt_3D(xyz):
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    xa, xd = pywt.dwt(x, wavelet)
    ya, yd = pywt.dwt(y, wavelet)
    za, zd = pywt.dwt(z, wavelet)
    return np.concatenate((xa[:, np.newaxis], ya[:, np.newaxis], za[:, np.newaxis],
                           xa[:, np.newaxis], ya[:, np.newaxis], zd[:, np.newaxis],
                           xa[:, np.newaxis], yd[:, np.newaxis], za[:, np.newaxis],
                           xa[:, np.newaxis], yd[:, np.newaxis], zd[:, np.newaxis],
                           xd[:, np.newaxis], ya[:, np.newaxis], za[:, np.newaxis],
                           xd[:, np.newaxis], ya[:, np.newaxis], zd[:, np.newaxis],
                           xd[:, np.newaxis], yd[:, np.newaxis], za[:, np.newaxis],
                           xd[:, np.newaxis], yd[:, np.newaxis], zd[:, np.newaxis]), axis=1)


def idwt_3D(dxyz):
    dxyz = dxyz.data.cpu().numpy()
    all_dxyz = []
    for i in range(dxyz.shape[0]):
        xa = (dxyz[i, :, 0] + dxyz[i, :, 3] + dxyz[i, :, 6] + dxyz[i, :, 9]) / 4.0
        xd = (dxyz[i, :, 12] + dxyz[i, :, 15] + dxyz[i, :, 18] + dxyz[i, :, 21]) / 4.0
        ya = (dxyz[i, :, 1] + dxyz[i, :, 4] + dxyz[i, :, 13] + dxyz[i, :, 16]) / 4.0
        yd = (dxyz[i, :, 7] + dxyz[i, :, 10] + dxyz[i, :, 19] + dxyz[i, :, 22]) / 4.0
        za = (dxyz[i, :, 2] + dxyz[i, :, 8] + dxyz[i, :, 14] + dxyz[i, :, 20]) / 4.0
        zd = (dxyz[i, :, 5] + dxyz[i, :, 11] + dxyz[i, :, 17] + dxyz[i, :, 23]) / 4.0
        x = pywt.idwt(xa, xd, wavelet)
        y = pywt.idwt(ya, yd, wavelet)
        z = pywt.idwt(za, zd, wavelet)
        xyz = np.concatenate((x[:, np.newaxis], y[:, np.newaxis], z[:, np.newaxis]), axis=1)
        all_dxyz.append(xyz[np.newaxis, ...])

    xyz = torch.Tensor(np.concatenate(all_dxyz)).to(device)
    return xyz
