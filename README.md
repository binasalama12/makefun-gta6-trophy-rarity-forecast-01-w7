[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/binasalama12/makefun-gta6-trophy-rarity-forecast-01-w7/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/binasalama12/makefun-gta6-trophy-rarity-forecast-01-w7/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

# gta6-trophy-rarity-forecast-01 — wallet w7

Forecast each of GTA VI's 55 PS5 trophies' Exophase earn rate (%) at the case
target, as five quantiles (5%, 25%, 50%, 75%, 95%) per trophy.

## Approach — power law in time

For each trophy, fit `logit(rate) = a + b * ln(hours since release + 1)` by
least squares over the last 24 hours of reads (release =
2026-11-19T00:00:00Z). Slopes are pooled per tier (median of the trophies'
slopes in that tier) and each trophy's slope is shrunk 50% toward its tier's
pooled slope, stabilising noisy fits for rare trophies.

The program branches on history variance:

- **Flat history** (every rate unchanged across the window — the launch-night
  pattern, where Exophase had not refreshed its counts): the page is stale
  and the next refresh dilutes every rate as new players flood in. Median =
  last rate × 0.32, spread = 1.3 logit units. The trophy already at 100% is
  exempt — a refresh left it at 100%, so it stays.
- **Trending history** (the final case, reads every 10 minutes for ~2 days):
  median = the shrunk power-law fit read at the target; spread =
  `0.5 * sqrt(horizon in days) + 0.3 + residual std` (residual std floored
  at 0.15), so longer horizons and noisier fits widen the band.

Quantiles use normal z-scores on the logit of the median, clamped to
[0.01, 99.99], rounded to 4 decimals (half up), forced non-decreasing.

## Run

```sh
MAKEFUN_INPUT_DIR=<input dir> MAKEFUN_OUTPUT_DIR=<output dir> python3 main.py
```

Reads every `*.in` in the input directory and writes `<id>.out` per case.
Python 3.13, standard library only, no network, no randomness — two runs
produce identical bytes. Runtime is far under the 8-second limit per case.

## Files

- `main.py` — the whole entry (single file, stdlib only).
- `README.md` — this file.
- `LICENSE` — MIT.
