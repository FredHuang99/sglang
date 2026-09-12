"""One nanosecond clock. Only serialization and profile coefficients use seconds."""
from decimal import Decimal, ROUND_HALF_EVEN

TICKS_PER_SECOND = 1_000_000_000
SEMANTICS = {"version": 3, "tick_seconds": 1e-9, "rounding": "half_even",
             "capacity_cache_sampling": "periodic_monitor_only", "arrival_mode": "equally_spaced",
             "startup_selection": "target_physical_modules"}


def ticks(value: float, *, positive: bool = False) -> int:
    if isinstance(value, bool):
        raise ValueError("Boolean is not a time")
    decimal = Decimal(str(value))
    if not decimal.is_finite():
        raise ValueError(f"Non-finite time: {value}")
    result = int((decimal * TICKS_PER_SECOND).to_integral_value(rounding=ROUND_HALF_EVEN))
    if positive and (decimal <= 0 or result <= 0):
        raise ValueError(f"Positive time must resolve to at least one nanosecond: {value}")
    return result


def seconds(value: int) -> float:
    return value / TICKS_PER_SECOND


def rounded_ratio(numerator: int, denominator: int) -> int:
    """Exact round-half-even, including rational arrival and step boundaries."""
    quotient, remainder = divmod(numerator, denominator)
    return quotient + int(2 * remainder > denominator or (2 * remainder == denominator and quotient % 2))


def elapsed(end_s: float, start_s: float) -> float:
    return seconds(ticks(end_s) - ticks(start_s))
