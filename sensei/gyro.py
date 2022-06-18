import numpy as np
import matplotlib.pyplot as plt


def compute_static_bias(bgs):
    a = -bgs
    b = bgs

    turn_on_bias = (b - a) * np.random.rand(1, 3) + a
    return turn_on_bias


def compute_white_noise(gyro_std):
    wn = np.ones(3)
    for i in range(0, 3):
        wn[i] *= np.random.normal(0, gyro_std)
    return wn


def compute_dynamic_bias(b_corr, b_dyn, prev_dbias, dt):
    # b_corr:  1x3 correlation times
    # b_dyn: 1x3 level of dynamic biases
    # dt: 1x1 sampling time

    # dbias_n: Simulated dynamic biases [X Y Z] (rad/s, rad/s, rad/s)
    dbias = np.zeros(3)

    for i in range(3):
        beta = dt / b_corr[i]
        sigma = b_dyn[i]
        a1 = np.exp(-beta)
        a2 = sigma * np.sqrt(1 - np.exp(-2 * beta))

        dbias[i] = a1 * prev_dbias[i] + a2 * np.random.normal(0, 1)
    return dbias


def compute_rate_random_walk_noise(rrw, prev_rrw_noise, dt):
    rrw_noise = np.zeros(3)
    for i in range(3):
        rrw_noise[i] = prev_rrw_noise[i] + dt * rrw * np.random.normal(0, 1)
    return rrw_noise


sb = compute_static_bias(np.ones(3) * 0.05)
wn = compute_white_noise(0.5)
