from typing import Dict

import pandas as pd

from ..enums import ActionType, PositionSide
from ..structs import StrategyStatus




class IchimokuStrategy:
    def __init__(self):
        pass

    def process(self, dataframes: Dict[str, pd.DataFrame]) -> StrategyStatus:
        dataframe_4h = dataframes["4h"]
        dataframe_1d = dataframes["1d"]
        ret = StrategyStatus()

        ha_close = dataframe_4h["ha_close"]
        senkou_span_a = dataframe_1d["SenkouSpanA"]
        senkou_span_b = dataframe_1d["SenkouSpanB"]
        cloud_green = senkou_span_a.iloc[-1] > senkou_span_b.iloc[-1]
        cloud_red = senkou_span_b.iloc[-1] > senkou_span_a.iloc[-1]
        if (
            (cloud_green and ha_close.iloc[-1] > senkou_span_a.iloc[-1] and ha_close.iloc[-2] < senkou_span_a.iloc[-1]) or \
            (cloud_red and ha_close.iloc[-1] > senkou_span_b.iloc[-1] and ha_close.iloc[-2] < senkou_span_b.iloc[-1])
        ):
            ret.action = ActionType.OpenLong

        ret.state = PositionSide.Short if (ha_close.iloc[-1] < senkou_span_a.iloc[-1] or ha_close.iloc[-1] < senkou_span_b.iloc[-1]) else PositionSide.Long

        return ret
