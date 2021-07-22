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


def compute_portfolio_return(weights, mean_returns):
    return weights.dot(mean_returns)

def compute_portfolio_variance(weights, covariance_matrix):
    return weights.dot(covariance_matrix).dot(weights)


class PortfolioOptimizer:
    
    def __init__(self, prices_df):
        """
        Parameters
        ----------
        prices_df : pandas dataFrame
            Prices dataframe with headers being symbols
        """
        pass
        