import math
from datetime import datetime

from beidou_spp.positioning.spp_solver import solve_epoch_spp


def test_spp_solver_converges_on_synthetic_geometry():
    receiver = (1_000_000.0, 2_000_000.0, 3_000_000.0)
    sats = {
        "C01": (20_200_000.0, 1_400_000.0, 21_700_000.0),
        "C02": (-18_600_000.0, 13_400_000.0, 20_100_000.0),
        "C03": (17_600_000.0, -21_200_000.0, 13_000_000.0),
        "C04": (-19_100_000.0, -14_800_000.0, 18_400_000.0),
        "C05": (23_200_000.0, 9_000_000.0, -12_100_000.0),
    }
    clock_bias = 75.0
    pseudoranges = {
        sat_id: math.dist(pos, receiver) + clock_bias for sat_id, pos in sats.items()
    }
    solution = solve_epoch_spp(
        sats,
        pseudoranges,
        epoch=datetime(2026, 4, 1),
        initial_position=(900_000.0, 2_100_000.0, 2_900_000.0),
        max_iter=15,
        threshold=1e-5,
    )
    assert solution.converged
    assert abs(solution.x_m - receiver[0]) < 1e-3
    assert abs(solution.receiver_clock_bias_m - clock_bias) < 1e-3


def test_spp_solver_fails_with_less_than_four_satellites():
    solution = solve_epoch_spp(
        {"C01": (20_000_000.0, 0.0, 0.0)},
        {"C01": 21_000_000.0},
        epoch=datetime(2026, 4, 1),
    )
    assert not solution.converged
    assert "少于 4" in solution.message

