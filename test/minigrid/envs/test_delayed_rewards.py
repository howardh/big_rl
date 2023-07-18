import pytest

from big_rl.minigrid.envs import RewardDelay


n_repeats = 100


##################################################
# Fixed Delay
##################################################


def test_no_delay():
    delay = RewardDelay(None)

    assert delay(0) == 0
    assert delay(1) == 1
    assert delay(12.3) == 12.3


def test_fixed_delay_0():
    delay = RewardDelay('fixed', steps=0)

    assert delay(1) == 1
    assert delay(12.3) == 12.3
    assert delay(0) == 0


def test_fixed_delay_1():
    delay = RewardDelay('fixed', steps=1)

    assert delay(1) == 0

    assert delay(12.3) == 1
    assert delay(0) == 12.3


def test_fixed_delay_3():
    delay = RewardDelay('fixed', steps=3)

    # 3 steps delay
    assert delay(1) == 0
    assert delay(2) == 0
    assert delay(3) == 0

    # Reward starts showing up
    assert delay(4) == 1
    assert delay(5) == 2

    # Reset
    delay.reset()

    # 3 steps delay
    assert delay(6) == 0
    assert delay(7) == 0
    assert delay(8) == 0

    # Reward starts showing up
    assert delay(9) == 6
    assert delay(10) == 7


##################################################
# Random Delay
##################################################

@pytest.mark.parametrize('overlap', ['replace', 'sum'])
def test_random_delay_0(overlap):
    delay = RewardDelay('random', steps=(0, 0), overlap=overlap)

    assert delay(1) == 1
    assert delay(12.3) == 12.3
    assert delay(0) == 0


@pytest.mark.parametrize('overlap', ['replace', 'sum'])
def test_random_delay_1(overlap):
    delay = RewardDelay('random', steps=(1, 1), overlap=overlap)

    assert delay(1) == 0

    assert delay(12.3) == 1
    assert delay(0) == 12.3


@pytest.mark.parametrize('overlap', ['replace', 'sum'])
def test_random_delay_3(overlap):
    delay = RewardDelay('random', steps=(3, 3), overlap=overlap)

    # 3 steps delay
    assert delay(1) == 0
    assert delay(2) == 0
    assert delay(3) == 0

    # Reward starts showing up
    assert delay(4) == 1
    assert delay(5) == 2

    # Reset
    delay.reset()

    # 3 steps delay
    assert delay(6) == 0
    assert delay(7) == 0
    assert delay(8) == 0

    # Reward starts showing up
    assert delay(9) == 6
    assert delay(10) == 7


def test_random_delay_0_to_1_replace():
    delay = RewardDelay('random', steps=(0, 1), overlap='replace')

    assert delay(1) in (0, 1)
    assert delay(12.3) in (0, 1, 12.3)
    assert delay(0) in (0, 12.3)


def test_random_delay_2_to_3_replace():
    delay = RewardDelay('random', steps=(2, 3), overlap='replace')

    for _ in range(n_repeats):
        delay.reset()

        # 2 steps delay
        assert delay(1) == 0
        assert delay(2) == 0

        # Reward starts showing up
        # Note: 0 is always possible because one reward could be delayed by 3 steps, then the next by 2 steps
        assert delay(3) in (0, 1)
        assert delay(4) in (0, 1, 2)
        assert delay(5) in (0, 2, 3)
        assert delay(6) in (0, 3, 4)


def test_random_delay_2_to_4_replace():
    delay = RewardDelay('random', steps=(2, 4), overlap='replace')

    for _ in range(n_repeats):
        delay.reset()

        # 2 steps delay
        assert delay(1) == 0
        assert delay(2) == 0

        # Reward starts showing up
        assert delay(3) in (0, 1)
        assert delay(4) in (0, 1, 2)
        assert delay(5) in (0, 1, 2, 3)
        assert delay(6) in (0, 2, 3, 4)
        assert delay(7) in (0, 3, 4, 5)
        assert delay(8) in (0, 4, 5, 6)


def test_random_delay_0_to_1_sum():
    delay = RewardDelay('random', steps=(0, 1), overlap='sum')

    assert delay(1) in (0, 1)
    for _ in range(n_repeats):
        assert delay(1) in (0, 1, 2)

    assert delay(2) in (0, 1, 2, 3)
    for _ in range(n_repeats):
        assert delay(2) in (0, 2, 4)


def test_random_delay_0_to_1_sum_clipped():
    delay = RewardDelay('random', steps=(0, 1), overlap='sum_clipped')

    assert delay(1) in (0, 1)
    for _ in range(n_repeats):
        assert delay(1) in (0, 1)

    for _ in range(n_repeats):
        assert delay(2) in (0, 1)

    assert delay(-2) in (0, 1, -1)
    for _ in range(n_repeats):
        assert delay(-2) in (0, -1)


##################################################
# Interval Delay
##################################################


@pytest.mark.parametrize('overlap', ['replace', 'sum', 'sum_clipped'])
def test_interval_delay_0(overlap):
    with pytest.raises(Exception):
        RewardDelay('interval', steps=0, overlap=overlap)


@pytest.mark.parametrize('overlap', ['replace', 'sum'])
def test_interval_delay_1(overlap):
    delay = RewardDelay('interval', steps=1, overlap=overlap)

    assert delay(1) == 1
    assert delay(12.3) == 12.3
    assert delay(0) == 0


@pytest.mark.parametrize('overlap', ['replace', 'sum'])
def test_interval_delay_1(overlap):
    delay = RewardDelay('interval', steps=1, overlap=overlap)

    assert delay(1) == 1
    assert delay(12.3) == 12.3
    assert delay(0) == 0


def test_interval_delay_2_replace():
    delay = RewardDelay('interval', steps=2, overlap='replace')

    assert delay(1) == 0
    assert delay(2) == 2
    assert delay(3) == 0
    assert delay(4) == 4
    assert delay(5) == 0
    assert delay(6) == 6


def test_interval_delay_3_replace():
    delay = RewardDelay('interval', steps=3, overlap='replace')

    assert delay(1) == 0
    assert delay(2) == 0
    assert delay(3) == 3
    assert delay(4) == 0
    assert delay(5) == 0
    assert delay(6) == 6


def test_interval_delay_2_sum():
    delay = RewardDelay('interval', steps=2, overlap='sum')

    assert delay(1) == 0
    assert delay(2) == 1 + 2
    assert delay(3) == 0
    assert delay(4) == 3 + 4
    assert delay(5) == 0
    assert delay(6) == 5 + 6


def test_interval_delay_3_sum():
    delay = RewardDelay('interval', steps=3, overlap='sum')

    assert delay(1) == 0
    assert delay(2) == 0
    assert delay(3) == 1 + 2 + 3
    assert delay(4) == 0
    assert delay(5) == 0
    assert delay(6) == 4 + 5 + 6


def test_interval_delay_3_sum_clipped():
    delay = RewardDelay('interval', steps=3, overlap='sum_clipped')

    # Positive sum
    assert delay(1) == 0
    assert delay(2) == 0
    assert delay(3) == 1

    assert delay(4) == 0
    assert delay(5) == 0
    assert delay(6) == 1

    # Zero sum
    assert delay(1) == 0
    assert delay(2) == 0
    assert delay(-3) == 0

    # Negative sum
    assert delay(-4) == 0
    assert delay(-5) == 0
    assert delay(6) == -1
