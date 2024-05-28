import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.graphics.regressionplots import plot_leverage_resid2, influence_plot
from statsmodels.stats.outliers_influence import variance_inflation_factor, OLSInfluence
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.model_selection import KFold
from scipy.stats import ttest_ind, shapiro, bartlett, levene, pearsonr, anderson, norm, zscore
from statsmodels.robust.norms import HuberT, Hampel
from statsmodels.robust.robust_linear_model import RLM
from linearmodels.panel import PanelOLS
import warnings
import patsy
import pingouin as pg