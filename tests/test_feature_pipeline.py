import pytest
import pandas as pd
import numpy as np
from src.indicators import SMA, FeaturePipeline, FeatureSpec, Diff, Lag, Rescale, ZScoreRolling

@pytest.fixture
def sample_ohlcv():
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    data = {
        "open": np.random.randn(100) + 100,
        "high": np.random.randn(100) + 105,
        "low": np.random.randn(100) + 95,
        "close": np.linspace(100, 110, 100) + np.random.randn(100), # Trend
        "volume": np.random.randint(100, 1000, 100)
    }
    return pd.DataFrame(data, index=dates)

def test_sma_basic(sample_ohlcv):
    period = 10
    sma = SMA(period)
    output = sma.compute(sample_ohlcv)
    
    assert isinstance(output, pd.DataFrame)
    assert len(output) == len(sample_ohlcv)
    assert output.columns == [f"sma_{period}_close"]
    
    # Check NaN for warmup
    assert output.iloc[period-2, 0] != output.iloc[period-2, 0] # NaN check (np.nan != np.nan) or pd.isna
    assert pd.isna(output.iloc[period-2, 0])
    assert not pd.isna(output.iloc[period-1, 0])

def test_pipeline_naming(sample_ohlcv):
    pipeline = FeaturePipeline([
        FeatureSpec(SMA(10), []),
        FeatureSpec(SMA(10), [Diff(1)]),
        FeatureSpec(SMA(10), [Diff(1), ZScoreRolling(20)]),
    ])
    
    X = pipeline.transform(sample_ohlcv)
    
    expected_cols = [
        "sma_10_close",
        "sma_10_close_diff_1",
        "sma_10_close_diff_1_zscore_20"
    ]
    
    assert list(X.columns) == sorted(expected_cols)

def test_pipeline_determinism(sample_ohlcv):
    pipeline = FeaturePipeline([
        FeatureSpec(SMA(10), [Diff(1)]),
        FeatureSpec(SMA(20), [Lag(1)]),
    ])
    
    run1 = pipeline.transform(sample_ohlcv)
    run2 = pipeline.transform(sample_ohlcv)
    
    pd.testing.assert_frame_equal(run1, run2)

def test_zscore_zeros(sample_ohlcv):
    # Test flat line where std = 0
    flat_data = pd.DataFrame({
        "open": [100]*50, "high": [100]*50, "low": [100]*50, "close": [100]*50, "volume": [100]*50
    }, index=pd.date_range("2023-01-01", periods=50))
    
    z = ZScoreRolling(10)
    out = z.apply(flat_data["close"].to_frame())
    
    # Std is 0, so zscore should be NaN or handled gracefully (we implemented NaN)
    assert out.iloc[20, 0] != out.iloc[20, 0] # Should be NaN

def test_max_lookback(sample_ohlcv):
    pipeline = FeaturePipeline([
        FeatureSpec(SMA(10), []), # lb=10
        FeatureSpec(SMA(10), [Diff(1)]), # lb=11
        FeatureSpec(SMA(10), [Diff(1), Lag(2)]), # lb=13
    ])
    
    assert pipeline.max_lookback == 13

def test_rescale(sample_ohlcv):
    rescale = Rescale(old_min=0, old_max=100, new_min=0, new_max=1)
    # Create valid data
    df = pd.DataFrame({"val": [0, 50, 100]}, index=[0, 1, 2])
    out = rescale.apply(df)
    
    assert out.iloc[0, 0] == 0.0
    assert out.iloc[1, 0] == 0.5
    assert out.iloc[2, 0] == 1.0

def test_pipeline_alias(sample_ohlcv):
    pipeline = FeaturePipeline([
        FeatureSpec(SMA(10), alias="my_custom_feature")
    ])
    X = pipeline.transform(sample_ohlcv)
    assert X.columns[0] == "my_custom_feature"

def test_input_validation(sample_ohlcv):
    bad_df = sample_ohlcv.drop(columns=["close"])
    pipeline = FeaturePipeline([FeatureSpec(SMA(10))])
    
    with pytest.raises(ValueError, match="missing required columns"):
        pipeline.transform(bad_df)

def test_custom_indicator_protocol_compliance(sample_ohlcv):
    from dataclasses import dataclass
    from src.indicators.core.interfaces import Indicator
    
    @dataclass(frozen=True)
    class MyCustomInd:
        p: int
        
        @property
        def name(self) -> str: return "custom"
        
        @property
        def lookback(self) -> int: return self.p
        
        def compute(self, df: pd.DataFrame) -> pd.DataFrame: return df[["close"]]

    custom = MyCustomInd(5)
    
    # Runtime check works because of @runtime_checkable
    assert isinstance(custom, Indicator)
    
    # Pipeline accepts it
    pipeline = FeaturePipeline([FeatureSpec(custom)])
    res = pipeline.transform(sample_ohlcv)
    assert not res.empty

