from nofx.trader import AutoTrader


if __name__ == "__main__":
    trader = AutoTrader(config_path="configs/default.yml")
    trader.run()
