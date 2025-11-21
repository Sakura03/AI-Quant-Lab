## 配置环境

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

## 更改参数配置
参数配置文件在目录 `configs` 下, 以下为配置文件的重要选项:
- `exchange:api_key`: 交易所的 API Key
- `exchange:secret`: 交易所的密钥
- `llm:api_key`: LLM的密钥 ([Deepseek 官网](https://platform.deepseek.com/))
- `logger:level_name`: 默认 `"INFO"`, 如果希望打印提示词, 设置为 `"DEBUG"`
- `prompt:template`: 使用的系统提示词模板
- `prompt:r_ratio`: 每个决策的最小盈亏比 (仅在提示词中提及该最小盈亏比, 在决策筛选中并未按照盈亏比筛选, 也就是说 LLM 可能给出盈亏比低于该值的决策)
- `prompt:max_positions`: 最大持仓数
- `prompt:altcoin_leverage`: 山寨币的最大杠杆
- `prompt:BTC_ETH_leverage`: BTC 和 ETH 的最大杠杆
- `prompt:history_span`: 每个时间周期给 LLM 多少个周期的历史数据
- `trader:mode`: 可选模式: `"live"` 和 `"backtest"`
- `trader:timeframe`: 交易周期 (每隔多长时间向 LLM 发出请求)
- `trader:symbols`: 交易对列表
- `indicators`: 技术指标 (见以下说明)

# 技术指标参数说明

参数 `indicators` 的格式是
```yaml
"timeframe": [...]
```
其中, `"timeframe"` 是时间周期, 如 `"15m"/"1h"/"4h"` 等, 表示给 LLM 哪些时间周期的数据, `[...]` 表示该时间周期下给 LLM 哪些技术指标

每个技术指标的格式如下:
```yaml
name: "EMA"
params:
  timeperiod: 7
display_name: "EMA7"
```
- `name`: 表示技术指标的名称, 现支持 `"SMA"/"EMA"/"BBANDS"/"MACD"/"RSI"/"ATR"/"KDJ"/"ICHIMOKU"`
- `params`: 技术指标的参数, 以 `"参数名": 参数值` 的字典的形式
- `display_name`: 技术指标传递给 LLM 时的名称, 如果该技术指标有多个值 (如 MACD), 该参数应为一个列表

# 回测模式专用参数
- `backtest:start_time`: 回测的开始时间
- `backtest:end_time`: 回测的结束时间
- `backtest:data_folder`: 回测数据所在目录

回测数据下载:
```bash
freqtrade download-data --exchange binance --timeframe 15m 1h 4h 1d --pairs BTC/USDT ETH/USDT SOL/USDT SUI/USDT --timerange 20240101-
```

将下载的数据移动到 `backtest:data_folder` 目录下

## 运行程序
```bash
python main.py
```

## 程序输出
- `logs` 目录: log 文件
- `results` 目录: 每个周期的账户/仓位/决策信息和可视化文件 (回测模式)
