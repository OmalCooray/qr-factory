# Quantitative Features Library

A modular, pure Python library for quantitative feature engineering.

## Architecture

The system is split into two layers:

1.  **Core (`src.indicators.core`)**: The engine. Contains interfaces (`Indicator`, `Transform`) and the `FeaturePipeline`. It is generic and dependency-free.
2.  **Implementation (`src.indicators.impl`)**: The standard library. Contains specific indicators (like SMA) and transforms (like Diff, Lag).

## Usage

### Basic Pipeline
Construct a `FeaturePipeline` using `FeatureSpec`s. Each spec connects a base indicator to a chain of transforms.

```python
from src.indicators import FeaturePipeline, FeatureSpec, SMA, Diff, ZScoreRolling

pipeline = FeaturePipeline([
    # Raw SMA(10)
    FeatureSpec(SMA(10)),
    
    # SMA(50) -> Diff(1)
    FeatureSpec(SMA(50), [Diff(1)]),
    
    # SMA(10) -> Diff(1) -> RollingZScore(20)
    FeatureSpec(SMA(10), [Diff(1), ZScoreRolling(20)]),
])

# Compute features (returns a DataFrame)
X = pipeline.transform(ohlcv_df)
```

## Extending the Library

### 1. Adding a Custom Indicator
Implement the `Indicator` protocol. You can use a dataclass or a regular class.

**Requirements:**
-   `name`: Unique string identifier.
-   `lookback`: Maximum periods required for warm-up.
-   `compute(ohlcv)`: Returns a DataFrame with the same index as input.

```python
from dataclasses import dataclass
import pandas as pd
from src.indicators import Indicator

@dataclass(frozen=True)
class RSI:
    period: int = 14
    
    @property
    def name(self) -> str:
        return f"rsi_{self.period}"

    @property
    def lookback(self) -> int:
        return self.period

    def compute(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        delta = ohlcv["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.to_frame(name=self.name)
```

### 2. Adding a Custom Transform
Implement the `Transform` protocol.

**Requirements:**
-   `name`: Unique string identifier (usually starting with `_`).
-   `lookback`: Periods consumed by the transform (e.g., lag needs k periods).
-   `apply(X)`: Returns a DataFrame with same shape and index.

```python
from dataclasses import dataclass
import pandas as pd
from src.indicators import Transform

@dataclass(frozen=True)
class AbsChange:
    @property
    def name(self) -> str:
        return "_abs"

    @property
    def lookback(self) -> int:
        return 0

    def apply(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.abs()
```

### 3. Usage
Once defined, plug them into the pipeline immediately:

```python
spec = FeatureSpec(base=RSI(14), transforms=[AbsChange()])
```
