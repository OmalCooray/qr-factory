from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ADX:
    period: int = 14

    @property
    def name(self) -> str:
        return f"adx_{self.period}"

    @property
    def lookback(self) -> int:
        return 2 * self.period

    def compute(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        for col in ("high", "low", "close"):
            if col not in ohlcv.columns:
                raise ValueError(f"Column '{col}' not found in input DataFrame.")

        high = ohlcv["high"].values.astype(np.float64)
        low = ohlcv["low"].values.astype(np.float64)
        close = ohlcv["close"].values.astype(np.float64)
        n = len(ohlcv)
        p = self.period

        # True Range, +DM, -DM
        tr = np.empty(n, dtype=np.float64)
        plus_dm = np.empty(n, dtype=np.float64)
        minus_dm = np.empty(n, dtype=np.float64)
        tr[0] = np.nan
        plus_dm[0] = np.nan
        minus_dm[0] = np.nan

        for i in range(1, n):
            h_diff = high[i] - high[i - 1]
            l_diff = low[i - 1] - low[i]
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1]),
            )
            plus_dm[i] = h_diff if (h_diff > l_diff and h_diff > 0) else 0.0
            minus_dm[i] = l_diff if (l_diff > h_diff and l_diff > 0) else 0.0

        # Wilder's smoothing for TR, +DM, -DM
        smoothed_tr = np.full(n, np.nan, dtype=np.float64)
        smoothed_plus = np.full(n, np.nan, dtype=np.float64)
        smoothed_minus = np.full(n, np.nan, dtype=np.float64)

        # First smoothed value = sum of first `period` valid values (indices 1..period)
        if n > p:
            smoothed_tr[p] = np.sum(tr[1 : p + 1])
            smoothed_plus[p] = np.sum(plus_dm[1 : p + 1])
            smoothed_minus[p] = np.sum(minus_dm[1 : p + 1])

            for i in range(p + 1, n):
                smoothed_tr[i] = smoothed_tr[i - 1] - smoothed_tr[i - 1] / p + tr[i]
                smoothed_plus[i] = (
                    smoothed_plus[i - 1] - smoothed_plus[i - 1] / p + plus_dm[i]
                )
                smoothed_minus[i] = (
                    smoothed_minus[i - 1] - smoothed_minus[i - 1] / p + minus_dm[i]
                )

        # +DI, -DI, DX
        plus_di = np.full(n, np.nan, dtype=np.float64)
        minus_di = np.full(n, np.nan, dtype=np.float64)
        dx = np.full(n, np.nan, dtype=np.float64)

        for i in range(p, n):
            if np.isnan(smoothed_tr[i]) or smoothed_tr[i] == 0:
                continue
            plus_di[i] = 100.0 * smoothed_plus[i] / smoothed_tr[i]
            minus_di[i] = 100.0 * smoothed_minus[i] / smoothed_tr[i]
            di_sum = plus_di[i] + minus_di[i]
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum if di_sum != 0 else 0.0

        # Wilder's smoothed ADX
        adx = np.full(n, np.nan, dtype=np.float64)
        first_adx_idx = 2 * p  # need `period` DX values starting at index `p`

        if n > first_adx_idx:
            # First ADX = mean of DX[p .. 2p-1]
            adx[first_adx_idx] = np.mean(dx[p + 1 : 2 * p + 1])
            for i in range(first_adx_idx + 1, n):
                adx[i] = (adx[i - 1] * (p - 1) + dx[i]) / p

        return pd.DataFrame({self.name: adx}, index=ohlcv.index)
