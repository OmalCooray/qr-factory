from dataclasses import dataclass, field
import pandas as pd
from typing import List, Optional
from .interfaces import Indicator, Transform, validate_ohlcv

@dataclass(frozen=True)
class FeatureSpec:
    base: Indicator
    transforms: List[Transform] = field(default_factory=list)
    alias: Optional[str] = None

    @property
    def name(self) -> str:
        if self.alias:
            return self.alias
        
        # Construct name: base_name + transform_names
        # Transform names are expected to start with "_"
        name = self.base.name
        for t in self.transforms:
            name += t.name
        return name

@dataclass(frozen=True)
class FeaturePipeline:
    specs: List[FeatureSpec]

    def transform(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """
        Computes features based on the pipeline specifications.
        
        Args:
            ohlcv: Input DataFrame with OHLCV data.
            
        Returns:
            DataFrame with all computed features, sorted alphabetically by column name.
        """
        validate_ohlcv(ohlcv)
        
        features = []
        for spec in self.specs:
            # 1. Compute base indicator
            df_feature = spec.base.compute(ohlcv)
            
            # 2. Apply transforms sequentially
            for t in spec.transforms:
                df_feature = t.apply(df_feature)
            
            # 3. Rename columns
            if len(df_feature.columns) == 1:
                df_feature.columns = [spec.name]
            
            features.append(df_feature)
        
        if not features:
            return pd.DataFrame(index=ohlcv.index)

        # 4. Concatenate
        X = pd.concat(features, axis=1)
        
        # 5. Sort columns alphabetically
        X = X.reindex(sorted(X.columns), axis=1)
        
        return X

    @property
    def max_lookback(self) -> int:
        max_lb = 0
        for spec in self.specs:
            lb = spec.base.lookback + sum(t.lookback for t in spec.transforms)
            if lb > max_lb:
                max_lb = lb
        return max_lb
