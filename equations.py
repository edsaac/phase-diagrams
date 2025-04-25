import numpy as np


def chemostat(Y, t, p):
    x, s = Y
    dxdt = x * s / (p[0] + s) - p[1] * x - p[2] * x
    dsdt = p[2] * (1 - s) - p[3] * x * s / (p[0] + s)
    return [dxdt, dsdt]


def logistic_growth(Y, t, p):
    x, s = Y
    dxdt = x * (1 - x) * s / (p[0] + s) - p[1] * x
    dsdt = p[2] * np.power(np.maximum(1 - x, 0), p[4]) * (1 - s) - p[3] * x * (1 - x) * s / (p[0] + s)
    return [dxdt, dsdt]


def adsorption_thomas(Y, t, p):
    c, q = Y
    Q_v, k_att, k_det, q_max = p

    site_block = 1 - q / q_max
    dcdt = Q_v * (1 - c) - k_att * c * site_block + k_det * q
    dqdt = k_att * c * site_block - k_det * q
    return [dcdt, dqdt]


def adsorption_second_order(Y, t, p):
    c, q = Y
    Q_v, k_att, k_det, q_max = p

    site_block = 1 - q / q_max
    dcdt = Q_v * (1 - c) - k_att * c * site_block + k_det * q * (q / q_max)
    dqdt = k_att * c * site_block - k_det * q * (q / q_max)
    return [dcdt, dqdt]


def adsorption_competition(Y, t, p):
    c0, c1, q0, q1 = Y

    Q_v, cinit_0, k_att0, k_det0, cinit_1, k_att1, k_det1, q_max = p

    ads = (q0 + q1) / q_max
    site_block = 1 - ads
    dc0dt = Q_v * (cinit_0 - c0) - k_att0 * c0 * site_block + k_det0 * q0
    dc1dt = Q_v * (cinit_1 - c1) - k_att1 * c1 * site_block + k_det1 * q1
    dq0dt = k_att0 * c0 * site_block - k_det0 * q0
    dq1dt = k_att1 * c1 * site_block - k_det1 * q1

    return [dc0dt, dc1dt, dq0dt, dq1dt]


def adsorption_triple_competition(Y, t, p):
    c0, c1, c2, q0, q1, q2 = Y

    Q_v, cinit_0, k_att0, k_det0, cinit_1, k_att1, k_det1, cinit_2, k_att2, k_det2, q_max = p

    ads = (q0 + q1 + q2) / q_max
    site_block = 1 - ads
    dc0dt = Q_v * (cinit_0 - c0) - k_att0 * c0 * site_block + k_det0 * q0
    dc1dt = Q_v * (cinit_1 - c1) - k_att1 * c1 * site_block + k_det1 * q1
    dc2dt = Q_v * (cinit_2 - c2) - k_att2 * c2 * site_block + k_det2 * q2

    dq0dt = k_att0 * c0 * site_block - k_det0 * q0
    dq1dt = k_att1 * c1 * site_block - k_det1 * q1
    dq2dt = k_att2 * c2 * site_block - k_det2 * q2

    return [dc0dt, dc1dt, dc2dt, dq0dt, dq1dt, dq2dt]
