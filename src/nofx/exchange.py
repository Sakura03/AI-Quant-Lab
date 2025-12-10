from typing import Tuple, List, Dict, Optional, Any

import logging
import time
import ccxt
import pandas as pd

from .logger import BaseClassWithLogger
from .enums import ActionType
from .structs import Balance, Position, SymbolData, MarketData, Action
from .utils import retry, format_symbol


class Exchange(BaseClassWithLogger):
    def __init__(self, name: str, api_key: str, secret: str, enable_rate_limit: bool = True, sandbox: bool = False, logger: Optional[logging.Logger] = None):
        super().__init__(logger=logger)

        params = {
            "apiKey": api_key,
            "secret": secret,
            "enableRateLimit": enable_rate_limit,  # 自动限速
            "options": {"defaultType": "future"},
        }

        if name == "binance":
            self.exchange = ccxt.binanceusdm(params)

            if sandbox:
                self.exchange.set_sandbox_mode(True)  # 切换到测试网
        else:
            raise NotImplementedError(f"未知的交易所: {name:s}")

        self.load_markets()

    """ CCXT Interface """
    @retry(max_retries=5, delay=5.0, raise_if_fail=True)
    def load_markets(self):
        self.exchange.load_markets()

    @retry(max_retries=5, delay=1.0, output=(0.001, 10.0))
    def fetch_order_restricts(self, symbol: str) -> Tuple[float, float]:
        """
            获取订单的最小交易数量 (以币为单位) 和最小名义价值
        """
        market = self.exchange.market(symbol)
        min_amount = market["limits"]["amount"]["min"]
        min_notional = market["limits"]["cost"]["min"]
        return (min_amount, min_notional)

    @retry(max_retries=5, delay=1.0, raise_if_fail=True)
    def fetch_balance(self, params: Dict[str, Any] = {}) -> Dict[str, Any]:
        return self.exchange.fetch_balance(params=params)

    @retry(max_retries=5, delay=2.0, output=[])
    def fetch_positions(self, symbols: Optional[List[str]] = None, params: Dict[str, Any] = {}) -> List[Dict[str, Any]]:
        return self.exchange.fetch_positions(symbols=symbols, params=params)

    @retry(max_retries=5, delay=2.0, output=[])
    def fetch_open_orders(self, symbol: Optional[str] = None, since: Optional[int] = None, limit: Optional[int] = None, params: Dict[str, Any] = {}) -> List[Dict[str, Any]]:
        return self.exchange.fetch_open_orders(symbol=symbol, since=since, limit=limit, params=params)

    @retry(max_retries=5, delay=5.0, output=[])
    def fetch_ohlcv(self, symbol: str, timeframe: str, since: Optional[int] = None, limit: Optional[int] = None, params: Dict[str, Any] = {}) -> List[Any]:
        return self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit, params=params)

    @retry(max_retries=5, delay=3.0, output=[])
    def fetch_funding_rate_history(self, symbol: str, since: Optional[int] = None, limit: Optional[int] = None, params: Dict[str, Any] = {}) -> List[Any]:
        return self.exchange.fetch_funding_rate_history(symbol, since=since, limit=limit, params=params)

    @retry(max_retries=5, delay=1.0, raise_if_fail=True)
    def fetch_mark_price(self, symbol: str) -> Optional[float]:
        return self.exchange.fetch_mark_price(symbol)["markPrice"]

    @retry(max_retries=5, delay=1.0)
    def fetch_open_interest(self, symbol: str) -> Optional[float]:
        return self.exchange.fetch_open_interest(symbol)["openInterestAmount"]

    @retry(max_retries=5, delay=1.0)
    def fetch_funding_rate(self, symbol: str) -> Optional[float]:
        return self.exchange.fetch_funding_rate(symbol)["fundingRate"]

    @retry(max_retries=5, delay=1.0, output=[])
    def cancel_all_orders(self, symbol: str, params: Dict[str, Any] = {}) -> List[Any]:
        return self.exchange.cancel_all_orders(symbol, params=params)

    @retry(max_retries=5, delay=1.0)
    def set_one_way_mode(self, symbol: str):
        position_mode = self.exchange.fetch_position_mode(symbol)
        if position_mode["hedged"]:
            self.exchange.set_position_mode(hedged=False, symbol=symbol)

    @retry(max_retries=5, delay=1.0, output={})
    def set_leverage(self, leverage: int, symbol: Optional[str] = None) -> Dict[str, Any]:
        return self.exchange.set_leverage(leverage=leverage, symbol=symbol)

    @retry(max_retries=5, delay=3.0, output={})
    def create_order(
            self,
            symbol: str,
            type: str,
            side: str,
            amount: Optional[float] = None,
            price: Optional[float] = None,
            params: Dict[str, Any] = {}
    ) -> Dict[str, Any]:
        position = self.get_position(symbol)
        if type in ["market", "limit"]:
            if params.get("reduceOnly", False):
                if not position:
                    self.warning(f"平仓失败: {symbol:s}的仓位不存在")
                    return {}
                amount = position.quantity
            else:
                if position:
                    self.warning(f"开仓失败: {symbol:s}的仓位已存在")
                    return {}

        if type == "limit":
            orders = self.exchange.fetch_open_orders(symbol)
            if any(order["type"].lower() == "limit" and order["side"].lower() == side.lower() for order in orders):
                self.warning(f"{symbol:s}的{side:s}方向{type:s}单已存在")
                return {}

        if type in ["STOP_MARKET", "TAKE_PROFIT_MARKET"]:
            orders = self.exchange.fetch_open_orders(symbol, params={"conditional": True})
            if any(order["info"]["orderType"].lower() == type.lower() and order["side"].lower() == side.lower() for order in orders):
                self.warning(f"{symbol:s}的{side:s}方向{type:s}单已存在")
                return {}

        return self.exchange.create_order(symbol, type, side, amount, price, params=params)

    """ Core function """
    def get_balance(self) -> Balance:
        balance = self.fetch_balance()
        return Balance.from_dict(balance)

    def get_position(self, symbol: str) -> Optional[Position]:
        positions = self.fetch_positions(symbols=[symbol], params={"useV2": True})

        if len(positions) > 1:
            self.warning(f"{symbol}有多个持仓, 对冲模式不允许")

        if len(positions) == 0:
            return None

        position = Position.from_dict(positions[0])
        return position if position.quantity != 0.0 else None

    def get_positions(self) -> List[Position]:
        positions = []
        for pos in self.fetch_positions(params={"useV2": True}):
            symbol = format_symbol(str(pos["symbol"]))
            stop_loss, take_profit = self.get_stop_loss_and_take_profit(symbol)
            position = Position.from_dict(pos, stop_loss=stop_loss, take_profit=take_profit)
            if position.quantity != 0.0:
                positions.append(position)

        return positions

    def get_stop_loss_and_take_profit(self, symbol: str) -> Tuple[Optional[float], Optional[float]]:
        orders = self.fetch_open_orders(symbol=symbol, params={"conditional": True})
        stop_loss, take_profit = None, None
        for order in orders:
            if order["type"].upper() == "STOP_MARKET":
                stop_loss = float(order["stopPrice"])
            elif order["type"].upper() == "TAKE_PROFIT_MARKET":
                take_profit = float(order["stopPrice"])
        return stop_loss, take_profit

    def fetch_ohlcv_df(self, symbol: str, timeframe: str, limit: int = 200) -> pd.DataFrame:
        ohlcv = self.fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df

    def fetch_history_ohlcv_df(self, symbol: str, timeframe: str, start_time: pd.Timestamp, end_time: pd.Timestamp) -> pd.DataFrame:
        all_candles = []
        since = start_time.value // int(1e6)
        until = end_time.value // int(1e6)
        limit = 200

        ms = since
        while True:
            candles = self.fetch_ohlcv(symbol, timeframe=timeframe, since=ms, limit=limit)
            if not candles:
                break

            all_candles.extend(candles)

            first_time = candles[0][0]
            last_time = candles[-1][0]
            first_time_str = pd.to_datetime(first_time, unit="ms").strftime("%Y-%m-%d %H:%M:%S")
            last_time_str = pd.to_datetime(last_time, unit="ms").strftime("%Y-%m-%d %H:%M:%S")
            self.info(f"[Fetch History OHLCV data][symbol: {symbol:s}][timeframe: {timeframe:s}][time range: {first_time_str:s} to {last_time_str:s}]")

            if last_time >= until:
                break

            ms = last_time + 1
            time.sleep(self.exchange.rateLimit / 1000)

        all_candles = [c for c in all_candles if since <= c[0] <= until]
        df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df

    def fetch_history_funding_rate(self, symbol: str, start_time: pd.Timestamp, end_time: pd.Timestamp) -> pd.DataFrame:
        all_funding = []
        since = start_time.value // int(1e6)
        until = end_time.value // int(1e6)
        limit = 200

        ms = since
        while True:
            data = self.fetch_funding_rate_history(symbol, since=ms, limit=limit)
            if not data:
                break

            for funding_rate in data:
                all_funding.append([funding_rate["timestamp"], funding_rate["fundingRate"]])

            first_time = data[0]["timestamp"]
            last_time = data[-1]["timestamp"]
            first_time_str = pd.to_datetime(first_time, unit="ms").strftime("%Y-%m-%d %H:%M:%S")
            last_time_str = pd.to_datetime(last_time, unit="ms").strftime("%Y-%m-%d %H:%M:%S")
            self.info(f"[Fetch History Funding Rate][symbol: {symbol:s}][time range: {first_time_str:s} to {last_time_str:s}]")

            if last_time >= until:
                break

            ms = last_time + 1
            time.sleep(self.exchange.rateLimit / 1000)

        all_funding = [d for d in all_funding if since <= d[0] <= until]
        df = pd.DataFrame(all_funding, columns=["timestamp", "fundingRate"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        # 去掉毫秒级的误差
        df["timestamp"] = df["timestamp"].dt.floor("s")
        return df

    def get_market_data(self, symbols: List[str], timeframes: List[str]) -> MarketData:
        market_data = dict()
        for symbol in symbols:
            mark_price = self.fetch_mark_price(symbol)
            if mark_price is None:
                continue

            market_data[symbol] = SymbolData(
                symbol=symbol,
                mark_price=mark_price,
                open_interest=self.fetch_open_interest(symbol),
                funding_rate=self.fetch_funding_rate(symbol),
                data={
                    timeframe: self.fetch_ohlcv_df(symbol, timeframe=timeframe)
                    for timeframe in timeframes
                }
            )
        return market_data

    def open_position(self, action: Action):
        assert action.type in [ActionType.OpenLong, ActionType.OpenShort]

        # cancel all open orders
        self.cancel_all_orders(action.symbol, params={"conditional": True})

        # set one-way mode
        self.set_one_way_mode(action.symbol)

        # set leverage
        self.set_leverage(leverage=action.leverage, symbol=action.symbol)

        # create market order
        # TODO(Xinyu): use limit order to save fee
        mark_price = self.fetch_mark_price(action.symbol)
        amount = action.position_size_usd / mark_price
        self.create_order(
            symbol=action.symbol,
            type="market",
            side="buy" if action.type == ActionType.OpenLong else "sell",
            amount=amount,
        )

        # create stop loss order
        self.create_order(
            symbol=action.symbol,
            type="STOP_MARKET",
            side="sell" if action.type == ActionType.OpenLong else "buy",
            params={
                "stopPrice": action.stop_loss,
                "closePosition": True
            },
        )

        # create take profit order
        self.create_order(
            symbol=action.symbol,
            type="TAKE_PROFIT_MARKET",
            side="sell" if action.type == ActionType.OpenLong else "buy",
            params={
                "stopPrice": action.take_profit,
                "closePosition": True
            },
        )

    def close_position(self, action: Action):
        assert action.type in [ActionType.CloseLong, ActionType.CloseShort]

        # close position
        self.create_order(
            symbol=action.symbol,
            type="market",
            side="sell" if action.type == ActionType.CloseLong else "buy",
            params={
                "reduceOnly": True,  # 仅平仓
            },
        )

        # cancel open orders (stop loss and take profit orders)
        self.cancel_all_orders(action.symbol, params={"conditional": True})

    def adjust_order(self, action: Action):
        assert action.type == ActionType.AdjustOrder

        # cancel all open orders
        self.cancel_all_orders(action.symbol, params={"conditional": True})

        # create stop loss order
        self.create_order(
            symbol=action.symbol,
            type="STOP_MARKET",
            side="sell" if action.stop_loss < action.take_profit else "buy",
            params={
                "stopPrice": action.stop_loss,
                "closePosition": True
            },
        )

        # create take profit order
        self.create_order(
            symbol=action.symbol,
            type="TAKE_PROFIT_MARKET",
            side="sell" if action.stop_loss < action.take_profit else "buy",
            params={
                "stopPrice": action.take_profit,
                "closePosition": True
            },
        )

    def execute_action(self, action: Action):
        if action.type in [ActionType.OpenLong, ActionType.OpenShort]:
            self.open_position(action)

        elif action.type in [ActionType.CloseLong, ActionType.CloseShort]:
            self.close_position(action)

        elif action.type == ActionType.AdjustOrder:
            self.adjust_order(action)

    def execute_actions(self, actions: List[Action]):
        for action in actions:
            self.execute_action(action)
