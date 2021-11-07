# portfolio_optimizer
A portfolio optimizer tool made with Python. It performs [Mean-Variance Optimization](./docs/LaiLec1.pdf) with historical data.

See [Modern portfolio theory](https://en.wikipedia.org/wiki/Modern_portfolio_theory)

## Requirements & Installation
* Python >=3.8

Install by:

``` bash
$ pip3 install -r requirements.txt
```

## Usage

This tool relies in two data sources for historical data: _Yahoo_ and _Quandl_

Input symbols of the assets to allocate as found in [_Yahoo_](https://finance.yahoo.com/lookup) and [_Quandl_](https://data.nasdaq.com/search?query=) into a .txt file.

An example of input to use Yahoo Finance can be found in [input_symbols.txt](./input_symbols.txt)

Use the executable script provided `optimize.py` with inputs:

``` bash
$ ./optimize.py -h
usage: optimize.py [-h] [-D DATA_SOURCE] [-s START_DATE] [-e END_DATE] [--allow_sorting] [-q] input_file output_path

A program to optimize a portfolio comprised by the provided list of symbols by using their historical data

positional arguments:
  input_file            Input .txt file with symbols to use in different lines
  output_path           Path to where to place output CSV files with optimized allocations

optional arguments:
  -h, --help            show this help message and exit
  -D DATA_SOURCE, --data_source DATA_SOURCE
                        Data source. Any of 'yahoo' or 'quandl'
  -s START_DATE, --start_date START_DATE
                        Start date: Begining of data to use for optimization with format 'YYYY/MM/DD'
  -e END_DATE, --end_date END_DATE
                        End date: End date of data to use for optimization with format 'YYYY/MM/DD'
  --allow_sorting       Whether to allow sorting or not
  -q, --quiet           Do not show graphics, only outputs

```

Outputs solutions will be saved as CSV on the provided directory for three different optimal allocations: Best sharpe ratio, best return and best volatility.

User can also, by using graphical interface, select a different optimal point along the [efficient frontier](https://en.wikipedia.org/wiki/Efficient_frontier) by cliking on such curve, obtaining such allocation solution into console and CSV.

See example:

``` bash
$ ./optimize.py input_symbols.txt . -s 2020/01/01 -e 2021/06/30


Solution for best_sharpe_ratio:
{
  "SPY": 0.8539378986813692,
  "^DJI": 0.0015900838428043996,
  "IYR": 0.0016390318486487758,
  "^VIX": 0.13421175262269405,
  "SH": 0.006932890869723934,
  "SDS": 0.001688342134748902
}
Expected annual return:  21.121205277147336 %
Expected annual return σ (volatility):  0.17278046597995164
Expected sharpe ratio:  1.1090464055877183


Solution for best_return:
{
  "SPY": 0.9999996474209317,
  "^DJI": 9.625618396819916e-08,
  "IYR": 7.123685863046004e-08,
  "^VIX": 1.8794937674548333e-07,
  "SH": 6.011200660602932e-09,
  "SDS": -1.769506409221165e-09
}
Expected annual return:  22.44009913406466 %
Expected annual return σ (volatility):  0.28696343097589666
Expected sharpe ratio:  0.7054966448321893


Solution for best_volatility:
{
  "SPY": 0.5060905779105721,
  "^DJI": 0.026647783540482174,
  "IYR": 0.011063457429010957,
  "^VIX": 0.0048751456313632134,
  "SH": 0.37749336086184754,
  "SDS": 0.07382967462672389
}
Expected annual return:  -5.072627841666288 %
Expected annual return σ (volatility):  0.014549273473526575
Expected sharpe ratio:  -3.5780542898353795
 Click somewhere on a line.
 Right-click to deselect.
 Annotations can be dragged.

```

![Solutions](assets/Figure_1.png)
![Cov Matrix](assets/Figure_2.png)
