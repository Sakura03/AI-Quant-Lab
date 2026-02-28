import sys

from trading.cli import main


if __name__ == "__main__":
    # Compatibility wrapper: allow `python -m trading.run_backtest --config ...`
    sys.argv.insert(1, "backtest")
    main()
