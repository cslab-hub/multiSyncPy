# %%
import numpy as np
import scipy.spatial
import scipy.signal
import scipy.stats

import multiSyncPy as msp
from multiSyncPy import synchrony_metrics as sm
from multiSyncPy import data_generation as dg


def get_basic_kuramoto_data():
    np.random.seed(42)

    kuramoto_args = {
        "K": 0.5,
        "phases": np.array([0, 0.4 * np.pi, 0.8 * np.pi, 1.2 * np.pi, 1.6 * np.pi]),
        "omegas": [1.0, 1.5, 2.0, 2.5, 3.0],
        "alpha": 0.5,
        "d_t": 0.01,
        "length": 1000,
    }

    return dg.kuramoto_data(**kuramoto_args)


def test_coherence():
    assert np.isclose(0.14933899432114464, sm.coherence_team(get_basic_kuramoto_data()))


def test_symbolic_entropy():
    assert np.isclose(3.714899711057103, sm.symbolic_entropy(get_basic_kuramoto_data()))


def test_rho():
    kuramoto_test_data_phases = np.angle(
        scipy.signal.hilbert(get_basic_kuramoto_data())
    )
    assert np.isclose(0.8149785584909867, sm.rho(kuramoto_test_data_phases)[1])


def test_rqa():
    recurrence_matrix = sm.recurrence_matrix(get_basic_kuramoto_data(), radius=0.5)

    assert np.isclose(
        (0.019321321321321323, 0.9725417055227438, 11.300936768149883, 999),
        sm.rqa_metrics(recurrence_matrix),
    ).all()


def test_kuramoto_weak_null():
    np.random.seed(42)

    kuramoto_test_data_sample = np.tile(
        get_basic_kuramoto_data(), (100, 1, 1)
    ) + np.random.normal(0, 0.1, (100, 5, 1000))

    kuramoto_test_data_sample = np.angle(
        scipy.signal.hilbert(kuramoto_test_data_sample)
    )

    assert np.isclose(
        (1.9527624051998052e-217, 1488.230058836042, 99),
        sm.kuramoto_weak_null(kuramoto_test_data_sample),
    ).all()


def test_symbolic_entropy_windowed():
    np.random.seed(42)

    kuramoto_args = {
        "K": 0.5,
        "phases": np.array([0, 0.4 * np.pi, 0.8 * np.pi, 1.2 * np.pi, 1.6 * np.pi]),
        "omegas": [1.0, 1.5, 2.0, 2.5, 3.0],
        "alpha": 0.5,
        "d_t": 0.01,
        "length": 10000,
    }

    kuramoto_test_data_long = dg.kuramoto_data(**kuramoto_args)

    pattern_entropy_windowed = sm.apply_windowed(
        kuramoto_test_data_long, sm.symbolic_entropy, 100
    )

    assert np.isclose(
        [2.41807725, 3.23318747, 3.00633523, 2.44619682, 2.14577759],
        pattern_entropy_windowed[:5],
    ).all()


def test_rqa_windowed():
    recurrence_func = lambda x: sm.rqa_metrics(sm.recurrence_matrix(x, radius=0.5))[0]

    recurrence_over_time = sm.apply_windowed(
        get_basic_kuramoto_data(), recurrence_func, window_length=75, step=25
    )

    assert np.isclose(
        [0.07423423, 0.06954955, 0.05225225, 0.03675676, 0.02486486],
        recurrence_over_time[:5],
    ).all()


def test_coherence_windowed():
    coherence_over_time = sm.apply_windowed(
        get_basic_kuramoto_data(), sm.coherence_team, window_length=75, step=25
    )

    assert np.isclose(
        [0.13941338, 0.14107468, 0.14243127, 0.15962883, 0.16160428],
        coherence_over_time[:5],
    ).all()


def test_rho_windowed():
    data_phases = np.angle(scipy.signal.hilbert(get_basic_kuramoto_data()))

    rho_func = lambda x: sm.rho(x)[1]

    rho_over_time = sm.apply_windowed(data_phases, rho_func, window_length=75, step=25)

    assert np.isclose(
        [0.9582254, 0.98922587, 0.98551124, 0.98389035, 0.97009667],
        rho_over_time[:5],
    ).all()
