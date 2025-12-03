"""Configuration values that are used throughout the package."""

LEVELS: list[int] = [1, 2, 6, 8, 11]

DEFAULT_FLAG_RADIUS: int = 3
EXPECTED_VISITING_ORDERS: dict[int, list[int] | list[list[int]]] = {
    1: [0],
    2: [0],
    6: [0, 1, 2],
    8: [
        [0, 1, 2],
        [0, 1, 0, 2],
    ],
    11: [1, 0, 1, 2],
}
LEVELS_REVERSE_FLAGS: list[int] = [6, 8, 11]
# NB: for levels 6, 8 and 11 I've tested the flags are in the reversed
# order, but I don't know if this will always hold

BOUNDARY_RADII: dict[int, dict[str, float]] = {
    1: {"rin": 1, "rout": 2},
    2: {"rin": 1, "rout": 2},
    6: {"rin": 1.5, "rout": 4},
    8: {"rin": 1.5, "rout": 4},
    11: {"rin": 1, "rout": 2},
}

ODMATS_AGE_WINDOW_HALF_SIZE: int = 5
ODMATS_WEIGHT_STD: float = 2.0
ODMATS_AGE_RANGE = (24, 80)
# NB: we have filtered ages outside the range [19, 98] so the window cannot be smaller than 24
# The upper bound of 80 is mainly for storage reasons, there are no clinical patients older than 80
