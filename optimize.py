#!/usr/bin/env python3
# coding: utf-8

import os
import warnings
import datetime
import json

import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize
from pandas_datareader import data
import mplcursors

np.set_printoptions(precision=3)


# Data
INPUT_SYMBOLS_FILEPATH = "input_symbols.txt"
SOURCE_ADJCLOSE_MAP = {"quandl": "AdjClose",
                       "yahoo": "Adj Close"}

DATA_SOURCE = "yahoo"
START_DATE = "2018/01/01"
END_DATE = "2019/12/31"

# Pipeline
NUM_SIMULATION_POINTS = 200
USE_LOG_RETURNS = False
PERCENTAGE_MULTPLIER = 1 if USE_LOG_RETURNS else 100
DATE_FORMAT = "%Y/%m/%d"
YEAR_TRADING_DAYS = 250

# Optimizer flags:
SHORTING_ALLOWED = True


# # 1. Get data using `pandas_datareader`
# See https://pandas-datareader.readthedocs.io/en/latest/remote_data.html#remote-data-quandl
# 
#  * [Quandl](https://pandas-datareader.readthedocs.io/en/latest/remote_data.html#quandl) `quandl`
#      * Daily financial data (prices of stocks, ETFs etc.) from Quandl. The symbol names consist of two parts: DB name and symbol name. DB names can be all the free ones listed on the Quandl website. Symbol names vary with DB name; for WIKI (US stocks), they are the common ticker symbols, in some other cases (such as FSE) they can be a bit strange. Some sources are also mapped to suitable ISO country codes in the dot suffix style shown above, currently available for [BE, CN, DE, FR, IN, JP, NL, PT, UK, US](https://www.quandl.com/search?query=).
#      * As of June 2017, each DB has a different data schema, the coverage in terms of time range is sometimes surprisingly small, and the data quality is not always good.
#      
#  * [Yahoo](https://pandas-datareader.readthedocs.io/en/latest/remote_data.html#quandl) `yahoo`:
#      * Retrieve daily stock prices (high, open, close, volu,e and adjusted close). Symbols as in [Yahoo Lookup](https://finance.yahoo.com/lookup)

with open(INPUT_SYMBOLS_FILEPATH) as file:
    symbols_names = [x.rstrip() for x in file.readlines()]

raw_df = data.DataReader(symbols_names, DATA_SOURCE,
                         start=START_DATE, end=END_DATE)

# Compute returns
for symbol in symbols_names:
    raw_df[("Return", symbol)] = raw_df[(SOURCE_ADJCLOSE_MAP[DATA_SOURCE], symbol)].pct_change()
    raw_df[("logReturn", symbol)] = raw_df[("Return", symbol)].apply(lambda x: np.log(1 + x))


# ## Get latest closing prices
_start_date = datetime.datetime.now() - datetime.timedelta(days=5)
_now_date = datetime.datetime.now()
latest_raw_df = data.DataReader(symbols_names, DATA_SOURCE,
                                start=_start_date.strftime(DATE_FORMAT),
                                end=_now_date.strftime(DATE_FORMAT))
latests_prices_ = latest_raw_df[SOURCE_ADJCLOSE_MAP[DATA_SOURCE]].to_numpy()

latests_prices = np.empty(latests_prices_.shape[1])
latests_prices[:] = np.nan
for i in range(latests_prices_.shape[1]):
    for x in latests_prices_[::-1, i]:
        if not np.isnan(x):
            latests_prices[i] = x


# # 2. Compute mean returns and Covariance matrix

true_returns = raw_df["Return"].to_numpy()
returns = raw_df["logReturn"] if USE_LOG_RETURNS else raw_df["Return"]
returns = returns.to_numpy()

mean_returns = np.nanmean(returns, axis=0)
true_mean_returns = np.nanmean(true_returns, axis=0)

# NaN logic
mask = np.isnan(mean_returns)
leave_out_symbols = np.array(symbols_names)[mask].tolist()
if leave_out_symbols:
    warnings.warn(f"Symbols {leave_out_symbols} will not be considered as no data avialable")
    for x in leave_out_symbols:
        symbols_names.remove(x)

returns = returns[:, ~mask]
mean_returns = mean_returns[~mask]
latests_prices = latests_prices[~mask]
true_returns = true_returns[:, ~mask]
true_mean_returns = true_mean_returns[~mask]

covariance_matrix = np.ma.cov(np.ma.masked_invalid(returns),
                              rowvar=False).filled(np.nan)
true_covariance_matrix = np.ma.cov(np.ma.masked_invalid(true_returns),
                                   rowvar=False).filled(np.nan)
    


# # 3. Portfolio and optimization

weights_0 = np.ones(mean_returns.shape[0]) / mean_returns.shape[0]
ones_vec = np.ones_like(weights_0)

def compute_portfolio_mean_return(weights, mean_returns):
    return weights.dot(mean_returns)

def compute_portfolio_variance(weights, covariance_matrix):
    return weights.dot(covariance_matrix).dot(weights)


# ## Optmization

# We aim to:
# $$
# w_{eff} = argmin (w^{T} \Sigma w)
# $$
# given
# 
# $ w^{T} \mu = \mu^{*}$ and $\sum{w} = 1$
# 
# and optionally:
#  * No shorting possible: $w > 0$
#  
# For Portfolio budget allocation:
#  * We have an `B` amount of € to allocate: $\sum{w_i P_i} <= B$ ?
#  * Discrete steps: $w_i // P_i == $ ? 

# constraintns
constrain_matrix = np.vstack((mean_returns, ones_vec))
bounds = None if SHORTING_ALLOWED else optimize.Bounds(np.zeros_like(ones_vec).T,
                                                       ones_vec.T * np.inf)

frontier_returns = []
frontier_volatilities = []
true_frontier_returns = []
true_frontier_volatilities = []
weights_allocations = []
objs_mus = np.linspace(mean_returns.min(), mean_returns.max(), 50)
for objective_mu in objs_mus:
    linear_constraints = optimize.LinearConstraint(constrain_matrix,
                                                   [objective_mu, 1], [objective_mu, 1])

    solution = optimize.minimize(compute_portfolio_variance, weights_0,
                                 args=(covariance_matrix * YEAR_TRADING_DAYS),
                                 constraints=[linear_constraints],
                                 bounds=bounds,
                                 method="trust-constr", jac="3-point",
                                 options={"xtol": 1e-8, "gtol": 1e-8})
    # Save solution
    weights_allocations.append(solution.x)

    portfolio_mean_return = compute_portfolio_mean_return(solution.x, mean_returns)
    frontier_returns.append(portfolio_mean_return * YEAR_TRADING_DAYS)

    true_portfolio_mean_return = compute_portfolio_mean_return(solution.x, true_mean_returns)
    true_frontier_returns.append(true_portfolio_mean_return * YEAR_TRADING_DAYS)

    portfolio_variance = compute_portfolio_variance(solution.x, covariance_matrix)
    portfolio_volatility = np.sqrt(portfolio_variance * YEAR_TRADING_DAYS)
    frontier_volatilities.append(portfolio_volatility)
    
    true_portfolio_variance = compute_portfolio_variance(solution.x, true_covariance_matrix)
    true_portfolio_volatility = np.sqrt(true_portfolio_variance * YEAR_TRADING_DAYS)
    true_frontier_volatilities.append(true_portfolio_volatility)

weights_allocations = np.array(weights_allocations)
frontier_returns = np.array(frontier_returns)
frontier_volatilities = np.array(frontier_volatilities)
true_frontier_returns = np.array(true_frontier_returns)
true_frontier_volatilities = np.array(true_frontier_volatilities)
sharpe_ratios = true_frontier_returns / true_frontier_volatilities

# Select solutions:
best_sharperatio_idx = sharpe_ratios.argmax()
best_volatility_idx = frontier_volatilities.argmin()
best_returns_idx = frontier_returns.argmax()

# ## Simulation
simulation_points = []
for i in range(weights_allocations.shape[0]):
    for _ in range(NUM_SIMULATION_POINTS):
        weights_ = weights_allocations[i, :] + np.random.rand(mean_returns.shape[0]) * 0.1
        weights_ /= weights_.sum()
        portfolio_mean_return = compute_portfolio_mean_return(weights_, mean_returns)
        simulation_points += [[np.sqrt(compute_portfolio_variance(weights_, covariance_matrix) * YEAR_TRADING_DAYS),
                               portfolio_mean_return * YEAR_TRADING_DAYS]]
simulation_points = np.array(simulation_points)
simulation_sharpe_ratios = simulation_points[:, 0] / simulation_points[:, 1]
simulation_sharpe_ratios_norm = np.clip(simulation_sharpe_ratios, -1, 1)


# ## Plot
fig = plt.figure(figsize=(10, 10))
ax = plt.gca()
plt.scatter(simulation_points[:, 0] * PERCENTAGE_MULTPLIER,
            simulation_points[:, 1] * PERCENTAGE_MULTPLIER,
            s=2, label="Random allocations",
            c=simulation_sharpe_ratios_norm, cmap="viridis")
frontier = plt.plot(frontier_volatilities * PERCENTAGE_MULTPLIER,
                    frontier_returns * PERCENTAGE_MULTPLIER,
                    label="Efficient frontier")

plt.plot(frontier_volatilities[best_sharperatio_idx] * PERCENTAGE_MULTPLIER,
         frontier_returns[best_sharperatio_idx] * PERCENTAGE_MULTPLIER, "*",
         markersize=20, label="Best Sharpe ratio allocation")

plt.plot(frontier_volatilities[best_volatility_idx] * PERCENTAGE_MULTPLIER,
         frontier_returns[best_volatility_idx] * PERCENTAGE_MULTPLIER, "*",
         markersize=20, label="Best volatily allocation")

plt.plot(frontier_volatilities[best_returns_idx] * PERCENTAGE_MULTPLIER,
         frontier_returns[best_returns_idx] * PERCENTAGE_MULTPLIER, "*",
         markersize=20, label="Best return allocation")

plt.title("Portfolio allocations")
if USE_LOG_RETURNS:
    plt.xlabel("Annualized volatility (standard deviation of log-returns)")
    plt.ylabel("Annualized log-returns")
else:
    plt.xlabel("Annualized volatility (standard deviation of returns)")
    plt.ylabel("Annualized returns (%)")

handles, labels = ax.get_legend_handles_labels()
labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
plt.legend(handles, labels)
print("Click somewhere on a line.\nRight-click to deselect.\n"
      "Annotations can be dragged.")
mplcursors.cursor()
plt.show()

# Correlations
volatility_matrix = covariance_matrix.copy() * YEAR_TRADING_DAYS

np.fill_diagonal(volatility_matrix, np.nan)

fig = plt.figure(figsize=(10, 10))
ax = fig.gca()
plot = ax.matshow(volatility_matrix, cmap="viridis")
for (i, j), z in np.ndenumerate(volatility_matrix):
    ax.text(j, i, "{:0.3f}".format(z), ha="center", va="center")
    
plt.xticks(range(0, len(symbols_names)), symbols_names, rotation=45);
plt.yticks(range(0, len(symbols_names)), symbols_names);
ax.set_title("Correlations matrix", fontsize=16)
fig.colorbar(plot)
plt.show()

# Output solutions
best_weights = weights_allocations[best_sharperatio_idx, :]

best_solution = dict(zip(symbols_names, best_weights))
best_solution_dump = json.dumps(best_solution, sort_keys=True, indent=4)
print("Solution:")
print(best_solution_dump)
expected_return = compute_portfolio_mean_return(best_weights, true_mean_returns) * YEAR_TRADING_DAYS * 100
print("Expected annual return (%): ", expected_return)
print("Expected sharpe ratio: ", sharpe_ratios[best_sharperatio_idx])
portfolio_variance = compute_portfolio_variance(best_weights, true_covariance_matrix)
portfolio_variance = np.sqrt(portfolio_variance * YEAR_TRADING_DAYS) * 100
print("Expected annual return σ (volatility): ", portfolio_variance)
print(f"Expected annual return ± band 68.3% certainty (% ± band): {expected_return} ± {portfolio_variance}")
print(f"Expected annual return ± band 95.5% certainty (% ± band): {expected_return} ± {2 * portfolio_variance}")
