"""Delivery task and executable checks for the PPO/SAC editorial.

Run with Python 3: python examples/delivery_checks.py
Standard library only. This verifies arithmetic, not learned performance.
"""
from decimal import Decimal as D
from math import atanh, exp, isclose, log, pi, sqrt
from random import Random


class Delivery:
    def __init__(self, seed=0):
        self.rng = Random(seed)
        self.reset()

    def reset(self):
        self.x, self.h, self.done = D('0'), 6, False
        return self.x, self.h

    def step(self, action, *, disturbance=None):
        """disturbance is a test hook; omit it for random environment steps."""
        if self.done:
            raise ValueError('Reset before stepping a terminated episode')
        a = D(str(action))
        if not a.is_finite() or not D('-1') <= a <= D('1'):
            raise ValueError('Action must be in [-1, 1]')
        eta = D(str(disturbance)) if disturbance is not None else self.rng.choice([D('-.1'), D('.1')])
        if eta not in (D('-.1'), D('.1')):
            raise ValueError('Disturbance must be -0.1 or +0.1')
        self.x = min(D('4'), max(D('0'), self.x + a + eta))
        self.h -= 1
        success = self.x >= D('3.8')
        reward = -D('1') - D('.1') * a * a + (D('10') if success else D('0'))
        self.done = success or self.h == 0
        return (self.x, self.h), reward, self.done


def check_examples():
    env = Delivery()
    rewards = []
    for eta, expected_x in zip(['.1', '.1', '.1', '-.1'], ['1.1', '2.2', '3.3', '4']):
        (x, h), r, done = env.step('1', disturbance=eta)
        assert x == D(expected_x)
        rewards.append(r)
    assert done and h == 2
    assert rewards == [D('-1.1')]*3 + [D('8.9')]
    assert sum(D('.9')**i * r for i, r in enumerate(rewards)) == D('3.5071')
    try:
        env.step(1)
    except ValueError:
        pass
    else:
        raise AssertionError('A finished episode must not pay another reward')

    for eta, next_x, reward, terminal in [('-0.1', '3.7', '-1.064', False), ('0.1', '3.9', '8.936', True)]:
        env.reset()
        env.x, env.h = D('3'), 2
        state, r, done = env.step('.8', disturbance=eta)
        assert state == (D(next_x), 1) and r == D(reward) and done == terminal
    q = D('.5') * (D('-1.064') + D('.9') * D('7')) + D('.5') * D('8.936')
    assert q == D('7.086')

    env.reset()
    env.x, env.h = D('2'), 3
    trajectory = [env.step('.8', disturbance='.1'), env.step('.8', disturbance='.1')]
    assert trajectory[0] == ((D('2.9'), 2), D('-1.064'), False)
    assert trajectory[1] == ((D('3.8'), 1), D('8.936'), True)
    delta0 = trajectory[0][1] + D('.9') * D('6') - D('4')
    delta1 = trajectory[1][1] - D('6')
    advantage0 = delta0 + D('.72') * delta1
    assert delta0 == D('.336') and delta1 == D('2.936')
    assert advantage0 == D('2.44992')
    assert advantage0 + D('4') == D('6.44992')
    for ratio, expected in [('1', '2.44992'), ('1.2', '2.939904'), ('1.5', '2.939904')]:
        ratio = D(ratio)
        clipped = min(D('1.2'), max(D('.8'), ratio))
        assert min(ratio * advantage0, clipped * advantage0) == D(expected)

    env.reset()
    env.x, env.h = D('2'), 1
    _, reward, done = env.step('0', disturbance='.1')
    assert done and reward == -1
    negative_a = reward - D('-.8')
    assert negative_a == D('-.2')
    assert min(D('.6')*negative_a, D('.8')*negative_a) == D('-.16')
    assert min(D('1.5')*negative_a, D('1.2')*negative_a) == D('-.3')

    assert isclose(atanh(.8), log(3))
    old_density = 1 / (sqrt(2*pi)*.36)
    new_density = 1 / (sqrt(2*pi)*(2/3)*.36)
    assert isclose(new_density/old_density, 1.5)
    target = D('-1.064') + D('.9') * (D('7') - D('.2')*D('-.5'))
    assert target == D('5.326')
    assert (D('5')-target)**2 == D('.106276')
    assert (D('6')-target)**2 == D('.454276')
    log_density = -.5*log(2*pi)-log(.36)
    assert isclose(log_density, log(old_density))
    assert isclose(.2*log_density-5, -4.979457457134538)
    print('PASS: trajectories, termination, Bellman, GAE, PPO clipping, density ratios and SAC losses')


if __name__ == '__main__':
    check_examples()
