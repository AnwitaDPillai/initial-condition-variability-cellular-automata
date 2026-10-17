import numpy as np
from src.automaton import evolve, random_initial
from src.features import rule_features


def test_rule_0_goes_to_zero():
    x = evolve(0, np.array([1, 0, 1, 1]), 4)
    assert np.all(x[1:] == 0)


def test_rule_255_goes_to_one():
    x = evolve(255, np.array([0, 0, 1, 0]), 3)
    assert np.all(x[1:] == 1)


def test_rule_204_copies_center():
    initial = np.array([0, 1, 1, 0, 1, 0], dtype=np.uint8)
    x = evolve(204, initial, 5)
    assert np.all(x == initial[None, :])


def test_steps_zero_keeps_initial():
    initial = np.array([0, 1, 0], dtype=np.uint8)
    x = evolve(30, initial, 0)
    assert x.shape == (1, 3) and np.array_equal(x[0], initial)


def test_same_seed_same_initial():
    assert np.array_equal(random_initial(200, 123), random_initial(200, 123))


def test_rule_90_xor_neighbors():
    initial = np.array([0, 0, 1, 0, 0], dtype=np.uint8)
    actual = evolve(90, initial, 1)[1]
    expected = np.bitwise_xor(np.roll(initial, 1), np.roll(initial, -1))
    assert np.array_equal(actual, expected)


def test_influences_in_range():
    for rule in range(256):
        f = rule_features(rule)
        assert 0 <= f['lambda'] <= 1
        assert all(0 <= f[k] <= 1 for k in ('influence_left','influence_center','influence_right'))


def test_lz78_constant_vs_alternating_and_random():
    from src.metrics import lz78_phrase_count
    constant = np.zeros(200, dtype=np.uint8)
    alternating = np.arange(200, dtype=np.uint8) % 2
    randomish = np.random.default_rng(44).integers(0, 2, 200, dtype=np.uint8)
    assert lz78_phrase_count(constant) < lz78_phrase_count(randomish)
    assert lz78_phrase_count(alternating) < lz78_phrase_count(randomish)


def test_symmetry_partition_has_88_classes():
    from src.symmetry import all_equivalence_classes
    groups = all_equivalence_classes()
    assert len(groups) == 256
    assert len(set(groups.values())) == 88
