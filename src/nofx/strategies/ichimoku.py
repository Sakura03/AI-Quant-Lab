from typing import Dict

import pandas as pd

from ..enums import ActionType, PositionSide
from ..structs import StrategyStatus
from ..utils import merge_timeframes


class IchimokuStrategy:
    def __init__(self):
        pass

    def process(self, dataframes: Dict[str, pd.DataFrame]) -> StrategyStatus:
        dataframe = merge_timeframes(dataframes["4h"], dataframes["1d"], suffix="_1d")

        ret = StrategyStatus()

        ha_close_4h = dataframe["ha_close"]
        senkou_a_1d = dataframe["SenkouSpanA_1d"] if "SenkouSpanA_1d" in dataframe.columns else dataframe["SenkouSpanA"]
        senkou_b_1d = dataframe["SenkouSpanB_1d"] if "SenkouSpanB_1d" in dataframe.columns else dataframe["SenkouSpanB"]
        cloud_green = senkou_a_1d.iloc[-1] > senkou_b_1d.iloc[-1]
        cloud_red = senkou_b_1d.iloc[-1] > senkou_a_1d.iloc[-1]
        if (
            (cloud_green and ha_close_4h.iloc[-1] > senkou_a_1d.iloc[-1] and ha_close_4h.iloc[-2] < senkou_a_1d.iloc[-2] and ha_close_4h.iloc[-2] < senkou_a_1d.iloc[-1]) or \
            (cloud_red and ha_close_4h.iloc[-1] > senkou_b_1d.iloc[-1] and ha_close_4h.iloc[-2] < senkou_b_1d.iloc[-2] and ha_close_4h.iloc[-2] < senkou_b_1d.iloc[-1])
        ):
            ret.action = ActionType.OpenLong

        ret.state = PositionSide.Short if (ha_close_4h.iloc[-1] < senkou_a_1d.iloc[-1] or ha_close_4h.iloc[-1] < senkou_b_1d.iloc[-1]) else PositionSide.Long

        return ret
