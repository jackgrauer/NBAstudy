import pandas as pd # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
import statsmodels.api as sm # type: ignore
import statsmodels.formula.api as smf # type: ignore
from statsmodels.graphics.regressionplots import plot_leverage_resid2, influence_plot # type: ignore
from statsmodels.stats.outliers_influence import variance_inflation_factor, OLSInfluence # type: ignore
from statsmodels.stats.diagnostic import het_breuschpagan # type: ignore
from statsmodels.stats.stattools import durbin_watson # type: ignore
from sklearn.linear_model import LinearRegression # type: ignore
from sklearn.metrics import mean_squared_error, explained_variance_score # type: ignore
from sklearn.model_selection import KFold # type: ignore
from scipy.stats import ttest_ind, shapiro, bartlett, levene, pearsonr, anderson, norm, zscore # type: ignore
from statsmodels.robust.norms import HuberT, Hampel # type: ignore
from statsmodels.robust.robust_linear_model import RLM # type: ignore
from linearmodels.panel import PanelOLS # type: ignore
import warnings # type: ignore
import patsy # type: ignore
import pingouin as pg # type: ignore