import torch
import torch.nn as nn
import numpy as np
import math


def dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """

    x_shape = x.shape
    # print("1",x.shape)
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)
    # print("2",x.shape)
    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)
    Vc = torch.fft.fft(v, dim=1)
    # print("3",Vc.shape)
    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc.real * W_r - Vc.imag * W_i

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)
    # print("4V",V.shape)
    return V


def idct(X, norm=None):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct(dct(x)) == x
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """
    # print("X",X.shape)
    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2
    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)
    V = torch.view_as_complex(V)

    v = torch.fft.ifft(V, dim=1).real
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]
    # print("xs",x.shape)
    return x.view(*x_shape)


class DCTAttention(nn.Module):
    def __init__(self):
        super().__init__()

        self.drop_attn = torch.nn.Dropout(p = 0.1)
        self.dct_percentage = 0.2

    def split_heads(self, X):
        X = X.reshape(X.size(0), X.size(1), self.num_head, self.head_dim)
        X = X.transpose(1, 2)
        return X

    def forward(self, X):
        # print('1',X.shape)

        B, T, C,W = X.shape
        X = X.view(B, T, -1)

        X = dct(X.transpose(1,2), norm='ortho').transpose(1,2)
        # print('2',X.shape)
        X = idct(X.transpose(1,2), norm='ortho').transpose(1,2)
        # print('3',X.shape)
        B2, T2, C2 = X.shape
        #short_X = X

        X = X.reshape(B2, T2, C, W)
        Q = X
        K = X
        V = X
        # mask is not applicable anymore

        dot = torch.matmul(Q, torch.transpose(K, -2, -1))
        dot = dot / math.sqrt(32)

        attn = nn.functional.softmax(dot, dim = -1)
        attn = self.drop_attn(attn)

        X = torch.matmul(attn, V)

        return X
