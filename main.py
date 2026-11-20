"""gta6-trophy-rarity-forecast-01 entry (wallet w7): power-law-in-time forecast.

Model
-----
For each trophy, fit logit(rate) = a + b * ln(hours since release + 1) by least
squares over the case's last 24 hours of reads (release = 2026-11-19T00:00:00Z).
The log-time regressor is the power-law view of trophy earning: early on, the
share of tracked players who have earned a trophy grows fast as the first
wave plays, then flattens as the tracked base keeps widening. Slopes are
pooled per tier (median of the trophies' slopes in that tier) and each trophy's
slope is shrunk 50% toward its tier's pooled slope, which stabilises the noisy
fits of rare trophies while letting common ones keep their own trend.

Two regimes, branched on history variance:

* Flat history (every rate unchanged across the window -- the launch-night
  pattern, where Exophase had not refreshed its counts): the page is stale and
  the next refresh dilutes every rate as fresh players flood in, so the median
  is the last rate times 0.32 and the spread is a wide 1.3 logit units. The
  one trophy already at 100% is exempt: a refresh left it at 100%, so it stays.
* Trending history (the final case, reads every 10 minutes for ~2 days): the
  median is the shrunk power-law fit read at the target, and the spread is
  0.5 * sqrt(horizon in days) + 0.3 + the fit's residual standard deviation
  (floored at 0.15), so longer horizons and noisier fits both widen the band.

Quantiles come from normal z-scores on the logit of the median, then are
clamped to [0.01, 99.99], rounded to 4 decimals (half up) and forced
non-decreasing.

Runtime: Python 3.13, standard library only, no network, no randomness --
two runs produce identical bytes.
"""

import json
import math
import os
import statistics
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# The grader's -P leaves the entry directory OFF sys.path: put it back first,
# so a module shipped beside this file imports.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# Dilution branch (flat history): median multiplier and logit spread.
FLAT_MEDIAN_FACTOR = 0.32
FLAT_SPREAD = 1.3
# Trend branch: random-walk sigma per sqrt-day, base spread, residual floor.
TREND_SIGMA = 0.5
TREND_BASE_SPREAD = 0.3
RESIDUAL_FLOOR = 0.15
# How far back the trend fit looks.
FIT_WINDOW = timedelta(hours=24)
# Shrinkage of each trophy's slope toward its tier's pooled slope.
SHRINK = 0.5


def parse_time(text: str) -> datetime:
    """Parse an ISO 8601 timestamp, accepting a trailing Z."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def clamp(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp(pct)
    return math.log(p / (100.0 - p))


def sigmoid(lg: float) -> float:
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x: float):
    """Round to 4 decimals, half up (as the house starter does).

    A whole number comes back as an int, so json.dumps writes 100, not 100.0.
    """
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def norm_ppf(p: float) -> float:
    """Standard normal quantile (Acklam's rational approximation)."""
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def log_hours(at: datetime) -> float:
    """ln(hours since release + 1), floored so pre-release reads stay valid."""
    hours = (at - RELEASE).total_seconds() / 3600.0
    return math.log(max(1.0 + hours, 0.05))


def least_squares(xs, ys):
    """Fit y = a + b*x; return (a, b, residual std). Degenerate input -> b=0."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if n < 2 or sxx <= 1e-12:
        return mean_y, 0.0, 0.0
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = mean_y - b * mean_x
    if n > 2:
        resid = math.sqrt(sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / (n - 2))
    else:
        resid = 0.0
    return a, b, resid


def last_non_null(reads, i):
    """Most recent non-null rate for trophy i, else 0.5 as a neutral fallback."""
    for read in reversed(reads):
        rate = read["rates"][i]
        if rate is not None:
            return rate
    return 0.5


def forecast_case(case: dict, name: str) -> dict:
    history = case["history"]
    if not history:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    levels = case["levels"]
    z_scores = [norm_ppf(level) for level in levels]
    checkpoint = parse_time(case["at"])
    target = parse_time(case["target"])
    horizon_days = case["horizon_minutes"] / 1440.0

    # Fit window: the last 24 hours of reads (all of them on the practice).
    window_start = checkpoint - FIT_WINDOW
    window = [r for r in history if parse_time(r["at"]) >= window_start]
    if not window:
        window = history

    # Flat history? Every trophy's rate unchanged across the whole window.
    flat = True
    for i in range(len(trophies)):
        seen = {r["rates"][i] for r in window if r["rates"][i] is not None}
        if len(seen) > 1:
            flat = False
            break

    x_target = log_hours(target)

    # Trend branch fits, computed once (also feeds the flat check's fallback).
    fits = []  # per trophy: (a, b, residual std)
    for i in range(len(trophies)):
        xs, ys = [], []
        for read in window:
            rate = read["rates"][i]
            if rate is not None:
                xs.append(log_hours(parse_time(read["at"])))
                ys.append(logit(rate))
        fits.append(least_squares(xs, ys))

    # Pool each tier's slopes (median), then shrink every slope toward it.
    pooled = {}
    for tier in {t["tier"] for t in trophies}:
        slopes = [fits[i][1] for i, t in enumerate(trophies) if t["tier"] == tier]
        pooled[tier] = statistics.median(slopes) if slopes else 0.0

    rows = []
    for i, trophy in enumerate(trophies):
        last = last_non_null(window, i)
        a, b, resid = fits[i]
        if flat:
            if last >= 99.99:
                median = last  # the 100% trophy survived the refresh untouched
            else:
                median = last * FLAT_MEDIAN_FACTOR
            spread = FLAT_SPREAD
        else:
            b_shrunk = b + SHRINK * (pooled[trophy["tier"]] - b)
            median = sigmoid(a + b_shrunk * x_target)
            spread = TREND_SIGMA * math.sqrt(horizon_days) + TREND_BASE_SPREAD \
                + max(resid, RESIDUAL_FLOOR)
        med_logit = logit(median)
        qs = []
        prev = None
        for z in z_scores:
            q = round4(clamp(sigmoid(med_logit + z * spread)))
            if prev is not None and q < prev:
                q = prev
            qs.append(q)
            prev = q
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
