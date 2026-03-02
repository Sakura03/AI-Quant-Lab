# 配置环境

1. 使用 conda 环境: 运行
```bash
conda env create --name nofx -f environment.yml
conda activate nofx
pip install .
```

2. 使用其它虚拟环境: 在干净的开发环境中运行
```bash
pip install -r requirements.txt
pip install .
```

# 更改参数配置
参数配置文件在目录 `configs` 下, 以下为配置文件的重要选项:
- `exchange:api_key`: 交易所的 API Key.
- `exchange:secret`: 交易所的密钥.
- `llm:api_key`: LLM的密钥 ([Deepseek 官网](https://platform.deepseek.com/)).
- `logger:level_name`: 默认 `"INFO"`, 如果希望打印提示词, 设置为 `"DEBUG"`.
- `prompt:template`: 使用的系统提示词模板.
- `prompt:r_ratio`: 每个决策的最小盈亏比 (仅在提示词中提及该最小盈亏比, 在决策筛选中并未按照盈亏比筛选, 也就是说 LLM 可能给出盈亏比低于该值的决策).
- `prompt:max_positions`: 最大持仓数.
- `prompt:altcoin_leverage`: 山寨币的最大杠杆.
- `prompt:BTC_ETH_leverage`: BTC 和 ETH 的最大杠杆.
- `prompt:history_span`: 每个时间周期给 LLM 多少个周期的历史数据.
- `trader:mode`: 可选模式: `"live"` 和 `"backtest"`.
- `trader:timeframe`: 交易周期 (每隔多长时间向 LLM 发出请求).
- `trader:symbols`: 交易对列表.
- `indicators`: 技术指标 (见以下说明).

## 技术指标参数说明

参数 `indicators` 的格式是
```yaml
"timeframe": [...]
```
其中, `"timeframe"` 是时间周期, 如 `"15m"/"1h"/"4h"` 等, 表示给 LLM 哪些时间周期的数据, `[...]` 表示该时间周期下给 LLM 哪些技术指标.

每个技术指标的格式如下:
```yaml
name: "EMA"
params:
  timeperiod: 7
display_name: "EMA7"
```
- `name`: 表示技术指标的名称, 现支持 `"SMA"`/`"EMA"`/`"BBANDS"`/`"MACD"`/`"RSI"`/`"ATR"`/`"KDJ"`/`"ICHIMOKU"`.
- `params`: 技术指标的参数, 以 `"参数名": 参数值` 的字典的形式.
- `display_name`: 技术指标传递给 LLM 时的名称, 如果该技术指标有多个值 (如 MACD), 该参数应为一个列表.

## 回测模式专用参数
- `backtest:start_time`: 回测的开始时间.
- `backtest:end_time`: 回测的结束时间.
- `backtest:data_folder`: 回测数据所在目录.

回测数据下载:
```bash
python generate_data.py configs/default.yml [--start-time 20240101-000000] [--end-time 20250701-000000] [--symbols BTC/USDT ETH/USDT] [--timeframes 1m 5m 1h]
```

可选参数: `--start-time`/`--end-time`/`--symbols`/`--timeframes`, 如果不指定参数, 则使用配置文件中的相应参数.

下载的数据位于目录 `backtest:data_folder`.

> 建议第一次下载时数据的时间段覆盖整个回测周期, 这样可以避免重复下载数据.

# 运行程序
```bash
python main.py configs/default.yml
```

## 程序输出
- `logs` 目录: log 文件.
- `results` 目录: 每个子目录保存一次运行的输出, 其中, `config.yml` 为配置文件, `snapshots` 目录保存每个周期的账户/仓位/决策的快照信息, `viz` 目录保存可视化文件 (回测模式).

# 可视化账户和持仓信息
```bash
python visualize.py --snapshot-path results/default_20251118_221840/snapshots --symbols BTC/USDT ETH/USDT --save-path /path/to/image.png
```

# Enterprise Trading Framework
`src/trading` 为多策略、多周期、自进化 walk-forward 框架，统一采用 `timestamp=K线收盘时间` 语义。

## CLI 命令
查看帮助:
```bash
PYTHONPATH=src python -m trading --help
```

单次回测:
```bash
PYTHONPATH=src python -m trading backtest --config configs/trading_enterprise_smoke.yml
```

Walk-forward 演化优化:
```bash
PYTHONPATH=src python -m trading optimize --config configs/trading_enterprise_smoke.yml
```

极小样本 smoke（用于链路验证）:
```bash
PYTHONPATH=src python -m trading optimize --config configs/trading_enterprise_smoke_mini.yml
```

## 配置文件用途
- `configs/trading_enterprise.yml`: 正式大规模实验配置。时间跨度最长、标的最多、策略最全、优化预算最高，适合最终评估与报告。
- `configs/trading_enterprise_smoke.yml`: 常规烟雾测试配置。规模中等，适合日常功能回归与参数逻辑验证。
- `configs/trading_enterprise_smoke_fast.yml`: 快速迭代配置。缩小标的与时间窗口，trial 数较低，适合开发期高频试错。
- `configs/trading_enterprise_smoke_mini.yml`: 最小链路配置。预算和数据规模最小，适合快速确认“代码能跑通”。

推荐使用顺序:
1. 先跑 `trading_enterprise_smoke_mini.yml` 确认链路与环境正常。
2. 再跑 `trading_enterprise_smoke_fast.yml` 做快速迭代。
3. 再跑 `trading_enterprise_smoke.yml` 做较完整回归。
4. 最后跑 `trading_enterprise.yml` 做正式优化与最终评估。

运行中进度监控:
- `backtest` 会输出时间轴进度（百分比、当前时间、当前持仓数）。
- `optimize` 会输出窗口进度、trial 进度、当前最佳验证分数，以及每窗口 test 结果摘要。
- 如需静默模式可加 `--quiet`，例如:
```bash
PYTHONPATH=src python -m trading optimize --config configs/trading_enterprise_smoke.yml --quiet
```

## 如何使用 optimize 模式得到的最优参数进行回测
`optimize` 完成后会输出 `best_genome.yml`，其中包含:
- `signal_tf`
- `regime_tf`
- `strategy_ids`
- `strategy_params`（各策略参数）

### 1) 先运行优化
```bash
PYTHONPATH=src python -m trading optimize --config configs/trading_enterprise_smoke.yml
```

### 2) 找到最新优化目录
```bash
LATEST_OPT_DIR=$(ls -td results/trading_enterprise/optimize_* | head -n 1)
echo "$LATEST_OPT_DIR"
```

### 3) 用 `best_genome.yml` 直接回测
```bash
PYTHONPATH=src python -m trading backtest \
  --config configs/trading_enterprise_smoke.yml \
  --genome-file "$LATEST_OPT_DIR/best_genome.yml"
```

### 4) 用同一组最优参数在其它时间段复验（推荐）
```bash
PYTHONPATH=src python -m trading backtest \
  --config configs/trading_enterprise_smoke.yml \
  --genome-file "$LATEST_OPT_DIR/best_genome.yml" \
  --start 20240101-000000 \
  --end 20240601-000000
```

说明:
- `--genome-file` 会自动加载优化得到的周期、策略列表和策略参数。
- 你仍可用 `--signal-tf/--regime-tf/--strategies` 显式覆盖。

## 结果输出文件说明
结果默认写入 `results/trading_enterprise`，每次运行会创建一个新目录:
- `backtest_YYYYmmdd_HHMMSS`
- `optimize_YYYYmmdd_HHMMSS`

### backtest 目录文件
- `equity_curve.csv`: 权益曲线时间序列，含 `equity/cash/open_positions/gross_notional/turnover/drawdown/max_symbol_share`。
- `trade_list.csv`: 每笔已平仓交易明细，含 `signal_time/entry_time/exit_time/fees/slippage/funding_pnl`。
- `metrics.json`: 本次回测核心指标（Sharpe、MaxDD、年化收益、交易数等）。
- `strategy_bundle.yml`: 本次回测实际使用的 `signal_tf/regime_tf/strategy_ids/strategy_params`。
- `resolved_config.yml`: 本次运行实际解析后的完整配置（便于复现实验）。
- `equity_drawdown.png`: 资金曲线与回撤率的静态图（便于快速查看/分享）。
- `symbol_performance.csv`: 按币种聚合的交易统计（交易数、胜率、profit factor、总收益、平均持仓时长、费用/滑点等）。
- `symbol_contributions.html`: 资金曲线增量（`equity-initial_equity`）与各币种 cumulative contribution 的总览图。
- `symbol_contributions_candles.html`: 二联图。上半部分为 `equity-initial_equity` 与各币种 contribution；下半部分为选中币种 K 线、开平仓点、买卖箭头、连线与每笔 `pnl_pct`，通过按钮切换币种。

### optimize 目录文件
- `all_trials.csv`: 全部 trial 记录（每窗口每 trial 的参数与 train/val 指标、得分）。
- `all_trials.parquet`: 与 `all_trials.csv` 内容相同的 parquet 版本（写入成功时生成）。
- `window_summary.csv`: 每个 walk-forward 窗口最终入选基因及 train/val/test 指标汇总。
- `best_genome.yml`: 全流程选出的最优基因（可直接给 `backtest --genome-file`）。
- `stitched_test_equity.csv`: 所有 test 窗口按时间拼接后的 OOS 权益曲线。
- `stitched_test_trades.csv`: 所有 test 窗口拼接后的 OOS 交易明细。
- `stitched_test_metrics.json`: 拼接 OOS 的最终指标（最重要的总评估文件）。
- `regime_metrics.csv`: 拼接 OOS 在 `bull/bear/sideways` 分段下的指标。
- `overfit_diagnostics.csv`: 过拟合诊断，重点看 `train_sharpe` 与 `val_sharpe` 差异。
- `resolved_config.yml`: 本次优化实际解析后的完整配置。
