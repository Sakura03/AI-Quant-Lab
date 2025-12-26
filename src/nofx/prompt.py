from typing import Tuple, Optional

import os.path as osp
import logging
import numpy as np
import json

from .logger import BaseClassWithLogger
from .structs import Balance, Context
from .utils import format_timeframe


class PromptManager(BaseClassWithLogger):
    template_folder = "prompt_template"

    def __init__(self, template: str, max_positions: int, max_leverage: int, run_timeframe: str, logger: Optional[logging.Logger] = None):
        super().__init__(logger=logger)

        self.template = template
        self.max_positions = max_positions
        self.max_leverage = max_leverage
        self.run_timeframe = run_timeframe

        system_prompt_file = osp.join(self.template_folder, self.template + ".txt")
        if not osp.isfile(system_prompt_file):
            self.template = "default"
            system_prompt_file = osp.join(self.template_folder, "default.txt")
            self.warning("模板不存在, 使用default模板")
        assert osp.isfile(system_prompt_file)

        with open(system_prompt_file, "r", encoding="utf-8") as f:
            self.system_prompt = f.read()

    def generate_system_prompt(self, balance: Balance) -> str:
        return self.system_prompt

    def generate_user_prompt(self, ctx: Context) -> str:
        balance = ctx.balance
        positions = ctx.positions
        market_data = ctx.market_data
        performance = ctx.performance

        if ctx.num_cycle < 30 or performance.return_std == 0.0 or performance.sharpe_ratio == 0.0:
            stage = "COLD_START"
        elif performance.sharpe_ratio < 0.0:
            stage = "DEFENSIVE"
        elif performance.sharpe_ratio < 0.7:
            stage = "NORMAL"
        else:
            stage = "AGGRESSIVE"

        json_dict = {
            "meta": {
                "ts": ctx.current_time.strftime("%Y-%m-%d %H:%M:%S"),
                "cycle": self.run_timeframe,
            },
            "portfolio": {
                "equity_usd": balance.total_wallet_balance + balance.total_unrealized_profit,
                "available_margin_usd": balance.available_balance,
                "open_positions_count": len(positions),
            },
            "strategy_state": {
                "stage": stage,
                "sample_count": ctx.num_cycle,
                "sharpe": performance.sharpe_ratio,
                "return_std": performance.return_std,
            },
            "constraints": {
                "max_leverage": self.max_leverage,
                "max_open_positions": self.max_positions,
            },
            "symbols": [],
        }

        for symbol, symbol_data in market_data.items():
            data = symbol_data.data
            # 4h 数据
            klines_4h = data["4h"]
            close = klines_4h["close"].iloc[-1]
            ema20 = klines_4h["EMA20"].iloc[-1]
            ema20_prev = klines_4h["EMA20"].iloc[-4]
            ema50 = klines_4h["EMA50"].iloc[-1]
            atr14 = klines_4h["ATR14"].iloc[-1]
            ema20_slope = (ema20 - ema20_prev) / atr14
            h0 = klines_4h["MACDHIST"].iloc[-1]
            h1 = klines_4h["MACDHIST"].iloc[-2]
            h2 = klines_4h["MACDHIST"].iloc[-3]
            rsi14 = klines_4h["RSI14"].iloc[-1]
            bb_mid = klines_4h["BBANDS_MID"].iloc[-1]

            last_swing_high = None
            klines_4h["low_min"] = np.minimum.reduce([
                klines_4h["low"],
                klines_4h["low"].shift(1),
                klines_4h["low"].shift(2),
                klines_4h["low"].shift(-1),
                klines_4h["low"].shift(-2),
            ])
            is_swing_high = (
                (klines_4h["high"] > klines_4h["high"].shift(1)) & \
                (klines_4h["high"] > klines_4h["high"].shift(2)) & \
                (klines_4h["high"] > klines_4h["high"].shift(-1)) & \
                (klines_4h["high"] > klines_4h["high"].shift(-2))
            )
            klines_swing_high = klines_4h.loc[is_swing_high]
            if len(klines_swing_high) > 0:
                is_valid_swing_high = (
                    (klines_swing_high["high"] - klines_swing_high["low_min"] >= 0.5 * atr14) & \
                    (klines_swing_high["high"] < klines_swing_high["high"].shift(1))
                )
                klines_valid_swing_high = klines_swing_high.loc[is_valid_swing_high]
                if len(klines_valid_swing_high) > 0:
                    last_swing_high = klines_valid_swing_high["high"].iloc[-1]

            last_swing_low = None
            klines_4h["high_max"] = np.maximum.reduce([
                klines_4h["high"],
                klines_4h["high"].shift(1),
                klines_4h["high"].shift(2),
                klines_4h["high"].shift(-1),
                klines_4h["high"].shift(-2),
            ])
            is_swing_low = (
                (klines_4h["low"] > klines_4h["low"].shift(1)) & \
                (klines_4h["low"] > klines_4h["low"].shift(2)) & \
                (klines_4h["low"] > klines_4h["low"].shift(-1)) & \
                (klines_4h["low"] > klines_4h["low"].shift(-2))
            )
            klines_swing_low = klines_4h.loc[is_swing_low]
            if len(klines_swing_low) > 0:
                is_valid_swing_low = (
                    (klines_swing_low["high_max"] - klines_swing_low["low"] >= 0.5 * atr14) & \
                    (klines_swing_low["low"] > klines_swing_low["low"].shift(1))
                )
                klines_valid_swing_low = klines_swing_low.loc[is_valid_swing_low]
                if len(klines_valid_swing_low) > 0:
                    last_swing_low = klines_valid_swing_low["low"].iloc[-1]

            last_breakout_level = None
            candidate_resistance = last_swing_high or klines_4h["close"].iloc[-11:-1].max()
            if close - candidate_resistance > 0.3 * atr14:
                last_breakout_level = candidate_resistance

            last_breakdown_level = None
            candidate_support = last_swing_low or klines_4h["close"].iloc[-11:-1].min()
            if candidate_support - close > 0.3 * atr14:
                last_breakdown_level = candidate_support

            # 1h 数据
            klines_1h = data["1h"]
            close_1h = klines_1h["close"].iloc[-1]
            ema20_1h = klines_1h["EMA20"].iloc[-1]
            atr14_1h = klines_1h["ATR14"].iloc[-1]
            rsi14_1h = klines_1h["RSI14"].iloc[-1]
            rsi14_prev_1h = klines_1h["RSI14"].iloc[-2]
            highest_20_1h = klines_1h["high"].iloc[-21:-1].max()
            lowest_20_1h = klines_1h["low"].iloc[-21:-1].min()

            # 1d 数据
            klines_1d = data["1d"]
            close_1d = klines_1d["close"].iloc[-1]
            ema20_1d = klines_1d["EMA20"].iloc[-1]
            ema50_1d = klines_1d["EMA50"].iloc[-1]
            rsi14_1d = klines_1d["RSI14"].iloc[-1]
            macd_hist_1d = klines_1d["MACDHIST"].iloc[-1]

            ### 4h Direction ###
            h4_direction_long = 0
            h4_direction_short = 0
            if close > ema20:
                h4_direction_long += 1
            elif close < ema20:
                h4_direction_short += 1

            if ema20 > ema50:
                h4_direction_long += 1
            elif ema20 < ema50:
                h4_direction_short += 1

            if ema20_slope > 0.25:
                h4_direction_long += 1
            elif ema20_slope < -0.25:
                h4_direction_short += 1

            if h0 > 0.0:
                h4_direction_long += 1
            elif h0 < 0.0:
                h4_direction_short += 1

            best = max(h4_direction_long, h4_direction_short)
            diff = abs(h4_direction_long - h4_direction_short)
            min_strength = 2
            diff_threshold = 1
            if best < min_strength or diff < diff_threshold:
                best_direction = "FLAT"
                h4_direction = 0
            elif h4_direction_long > h4_direction_short:
                best_direction = "LONG"
                h4_direction = h4_direction_long
            else:
                best_direction = "SHORT"
                h4_direction = h4_direction_short

            ### 4h Momentum ###
            h4_momentum = 0
            macd_eps = 1e-8
            if (
                (best_direction == "LONG" and h0 > macd_eps) or \
                (best_direction == "SHORT" and h0 < -macd_eps)
            ):
                h4_momentum += 1

            rsi_eps = 2.0
            if (
                (best_direction == "LONG" and rsi14 > 50.0 + rsi_eps) or \
                (best_direction == "SHORT" and rsi14 < 50.0 - rsi_eps)
            ):
                h4_momentum += 1

            min_diff = max(abs(h2) * 0.05, macd_eps)
            if (
                (best_direction == "LONG" and h0 - h1 > min_diff and h1 - h2 > min_diff) or \
                (best_direction == "SHORT" and h1 - h0 > min_diff and h2 - h1 > min_diff)
            ):
                h4_momentum += 1

            ### 4h Structure ###
            ratio = 0.6
            near_ma = abs(close - ema20) < ratio * atr14 or abs(close - bb_mid) < ratio * atr14
            if best_direction == "LONG":
                structure_hold = last_swing_low is not None and close > last_swing_low
                pullback_ok = near_ma and structure_hold
                breakout_retest_ok = (
                    last_breakout_level is not None and \
                    abs(close - last_breakout_level) < ratio * atr14 and \
                    close > last_breakout_level
                )
                h4_structure = int(pullback_ok or breakout_retest_ok)
            elif best_direction == "SHORT":
                structure_hold = last_swing_high is not None and close < last_swing_high
                pullback_fail_ok = near_ma and structure_hold
                breakdown_retest_ok = (
                    last_breakdown_level is not None and \
                    abs(close - last_breakdown_level) < ratio * atr14 and \
                    close < last_breakdown_level
                )
                h4_structure = int(pullback_fail_ok or breakdown_retest_ok)
            else:
                h4_structure = 0

            ### 1d Direction ###
            multi_tf = int(
                (best_direction == "LONG" and ema20_1d > ema50_1d) or \
                (best_direction == "SHORT" and ema20_1d < ema50_1d)
            )

            daily_bias_score = 0
            if ema20_1d > ema50_1d:
                daily_bias_score += 2
            elif ema20_1d < ema50_1d:
                daily_bias_score -= 2

            if close_1d > ema20_1d:
                daily_bias_score += 1
            elif close_1d < ema20_1d:
                daily_bias_score -= 1

            if rsi14_1d > 55.0:
                daily_bias_score += 1
            elif rsi14_1d < 45.0:
                daily_bias_score -= 1

            if macd_hist_1d > 0.0:
                daily_bias_score += 1
            elif macd_hist_1d < 0.0:
                daily_bias_score -= 1

            daily_bias = int(np.sign(daily_bias_score) * abs(daily_bias_score) // 2)

            ### Trend Score ###
            trend_score = h4_direction + h4_momentum + h4_structure + multi_tf

            ### 1h Trigger ###
            trigger_long_T1 = (
                abs(close_1h - ema20_1h) < 0.5 * atr14_1h and \
                close_1h > ema20_1h and \
                rsi14_1h > 45.0 and \
                rsi14_1h > rsi14_prev_1h
            )

            trigger_long_T2 = (close_1h - highest_20_1h > 0.2 * atr14_1h)

            trigger_short_T1 = (
                abs(close_1h - ema20_1h) < 0.5 * atr14_1h and \
                close_1h < ema20_1h and \
                rsi14_1h < 55.0 and \
                rsi14_1h < rsi14_prev_1h
            )

            trigger_short_T2 = (lowest_20_1h - close_1h > 0.2 * atr14_1h)

            h1_trigger_ok = 0
            h1_trigger_type = "NONE"
            if best_direction == "LONG":
                h1_trigger_ok = int(trigger_long_T1 or trigger_long_T2)
                if trigger_long_T1:
                    h1_trigger_type = "T1_PULLBACK_CONFIRM"
                elif trigger_long_T2:
                    h1_trigger_type = "T2_BREAKOUT_CONFIRM"
            elif best_direction == "SHORT":
                h1_trigger_ok = int(trigger_short_T1 or trigger_short_T2)
                if trigger_short_T1:
                    h1_trigger_type = "T1_PULLBACK_CONFIRM"
                elif trigger_short_T2:
                    h1_trigger_type = "T2_BREAKOUT_CONFIRM"

            position = None
            for p in positions:
                if p.symbol == symbol:
                    position = p
                    break
            position_dict = {}
            if position is not None:
                position_dict["has_position"] = 1
                position_dict["side"] = position.side.name
                position_dict["entry_price"] = position.entry_price
                position_dict["position_size_usd"] = position.entry_price * position.quantity
                position_dict["leverage"] = position.leverage
                position_dict["stop_loss"] = position.stop_loss
                position_dict["take_profit"] = position.take_profit
                if position.entry_time:
                    position_dict["opened_at"] = position.entry_time.strftime("%Y-%m-%d %H:%M:%S")
                position_dict["unrealized_pnl_pct"] = position.unrealized_pnl_pct
            else:
                position_dict["has_position"] = 0

            symbol_dict = {
                "symbol": symbol,
                "price": symbol_data.mark_price,
                "position": position_dict,
                "signal": {
                    "trigger_ok": h1_trigger_ok,
                    "trigger_type": h1_trigger_type,
                    "direction": best_direction,
                    "trend_score": trend_score,
                    "h4_direction": h4_direction,
                    "h4_momentum": h4_momentum,
                    "h4_structure": h4_structure,
                    "multi_tf": multi_tf,
                    "daily_bias": daily_bias,
                    "atr_1h": atr14_1h,
                },
            }

            json_dict["symbols"].append(symbol_dict)

        json_str = json.dumps(json_dict, ensure_ascii=False, indent=2)
        user_prompt = "以下是当前的市场与策略状态 (JSON):\n```json\n" + json_str + "\n```"
        return user_prompt

    def __call__(self, ctx: Context) -> Tuple[str, str]:
        system_prompt = self.generate_system_prompt(ctx.balance)
        user_prompt = self.generate_user_prompt(ctx)
        return system_prompt, user_prompt
