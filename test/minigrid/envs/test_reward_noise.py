import numpy as np
import scipy.stats

from big_rl.minigrid.envs import RewardNoise


n_repeats = 100


def test_no_noise():
    noise = RewardNoise(None)

    assert noise(0) == 0
    assert noise(1) == 1
    assert noise(12.3) == 12.3


def test_zero_noise_p_1():
    # Always set to 0
    noise = RewardNoise('zero', 1.0)

    assert noise(0) == 0
    assert noise(1) == 0
    assert noise(12.3) == 0


def test_zero_noise_p_1():
    # Never set to 0
    noise = RewardNoise('zero', 0.0)

    assert noise(0) == 0
    assert noise(1) == 1
    assert noise(12.3) == 12.3


def test_zero_noise_50_50():
    # 50% chance to set to 0
    noise = RewardNoise('zero', 0.5)

    output = np.array([noise(1) for _ in range(n_repeats)])

    assert scipy.stats.binom_test(output.sum(), n=n_repeats, p=0.5) > 0.05


def test_gaussian_noise():
    # Make sure the data is normal
    noise = RewardNoise('gaussian', 1.0)

    output = [noise(0.5) for _ in range(n_repeats)]

    assert scipy.stats.normaltest(output).pvalue > 0.05


def test_stop_noise_never_zero():
    noise = RewardNoise('stop', 0.0)

    assert noise(0) == 0
    assert noise(1) == 1
    assert noise(12.3) == 12.3


def test_stop_noise_always_zero():
    noise = RewardNoise('stop', 1.0)

    assert noise(0) == 0
    assert noise(1) == 0
    assert noise(12.3) == 0


def test_stop_noise_3_steps():
    noise = RewardNoise('stop', 3)

    assert noise(1) == 1
    assert noise(2) == 2
    assert noise(3) == 3

    assert noise(4) == 0
    assert noise(5) == 0
    assert noise(6) == 0
