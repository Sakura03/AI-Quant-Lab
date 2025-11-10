from typing import List, Optional

import logging
import ccxt
import pandas as pd

from .logger import BaseClassWithLogger
from .enums import ActionType
from .structs import Balance, Position, SymbolData, MarketData, Action


class Exchange(BaseClassWithLogger):
    def __init__(self, name: str, api_key: str, secret: str, enable_rate_limit: bool = True, sandbox: bool = False, logger: Optional[logging.Logger] = None):
        super().__init__(logger=logger)

        params = {
            'apiKey': api_key,
            'secret': secret,
            'enableRateLimit': enable_rate_limit,  # 自动限速
            "options": {"defaultType": "future"},
        }

        if name == "binance":
            self.exchange = ccxt.binance(params)

            if sandbox:
                self.exchange.set_sandbox_mode(True)  # 切换到测试网
        else:
            raise NotImplementedError(f"未知的交易所: {name:s}")

    def get_balance(self) -> Balance:
        balance = self.exchange.fetch_balance()
        return Balance.from_dict(balance["info"])

    def get_position(self, symbol: str) -> Optional[Position]:
        positions = self.exchange.fetch_positions(symbols=[symbol], params={"useV2": True})
        if len(positions) > 1:
            self.warning(f"{symbol}有多个持仓, 对冲模式不允许")

        if len(positions) == 0:
            return None

        position = positions[0]
        return Position.from_dict(position) if float(position["contracts"]) != 0.0 else None

    def get_positions(self) -> List[Position]:
        positions = []
        for pos in self.exchange.fetch_positions(params={"useV2": True}):
            if float(pos["contracts"]) != 0.0:
                positions.append(Position.from_dict(pos))

        return positions

    def fetch_ohlcv_df(self, symbol: str, timeframe: str, limit: int = 1000) -> pd.DataFrame:
        ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df

    def fetch_mark_price(self, symbol: str) -> float:
        mp = self.exchange.fetch_mark_price(symbol)
        assert "markPrice" in mp
        return mp["markPrice"]

    def fetch_open_interest(self, symbol: str) -> Optional[float]:
        oi = self.exchange.fetch_open_interest(symbol)
        return oi["openInterestAmount"] if "openInterestAmount" in oi else None

    def fetch_funding_rate(self, symbol: str) -> Optional[float]:
        fr = self.exchange.fetch_funding_rate(symbol)
        return fr["fundingRate"]

    def get_market_data(self, symbols: List[str], timeframes: List[str]) -> MarketData:
        market_data = dict()
        for symbol in symbols:
            market_data[symbol] = SymbolData(
                symbol=symbol,
                mark_price=self.fetch_mark_price(symbol),
                open_interest=self.fetch_open_interest(symbol),
                funding_rate=self.fetch_funding_rate(symbol),
                data={
                    timeframe: self.fetch_ohlcv_df(symbol, timeframe=timeframe)
                    for timeframe in timeframes
                }
            )
        return market_data

    def execute_action(self, action: Action):
        if action.type in [ActionType.OpenLong, ActionType.OpenShort]:
            # cancel all open orders
            self.exchange.cancel_all_orders(action.symbol)

            # fetch current position
            position = self.get_position(action.symbol)
            if position:
                self.warning(f"开仓失败: {action.symbol}的仓位已存在")
                return

            # set one-way mode
            position_mode = self.exchange.fetch_position_mode(action.symbol)
            if position_mode["hedged"]:
                self.exchange.set_position_mode(hedged=False, symbol=action.symbol)

            # set leverage
            self.exchange.set_leverage(leverage=action.leverage, symbol=action.symbol)

            # create market order
            # TODO(Xinyu): use limit order to save fee
            mark_price = self.fetch_mark_price(action.symbol)
            amount = action.position_size_usd / mark_price
            self.exchange.create_order(
                symbol=action.symbol,
                type="market",
                side="buy" if action.type == ActionType.OpenLong else "sell",
                amount=amount,
            )

            # stop loss order
            self.exchange.create_order(
                symbol=action.symbol,
                type="STOP_MARKET",
                side="sell" if action.type == ActionType.OpenLong else "buy",
                amount=None,
                params={
                    "stopPrice": action.stop_loss,
                    "closePosition": True
                },
            )

            # take profit order
            self.exchange.create_order(
                symbol=action.symbol,
                type="TAKE_PROFIT_MARKET",
                side="sell" if action.type == ActionType.OpenLong else "buy",
                amount=None,
                params={
                    "stopPrice": action.take_profit,
                    "closePosition": True
                },
            )

        elif action.type in [ActionType.CloseLong, ActionType.CloseShort]:
            # fetch current position
            position = self.get_position(action.symbol)
            if position is None:
                self.warning(f"平仓失败: {action.symbol}的仓位不存在")
                return

            # close position
            self.exchange.create_order(
                symbol=action.symbol,
                type="market",
                side="sell" if action.type == ActionType.CloseLong else "buy",
                amount=position.quantity,
                params={
                    "reduceOnly": True,  # 仅平仓
                },
            )

            # cancel open orders (stop loss and take profit orders)
            self.exchange.cancel_all_orders(action.symbol)

    def execute_actions(self, actions: List[Action]):
        for action in actions:
            self.execute_action(action)
