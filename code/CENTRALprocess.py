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
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error, explained_variance_score # type: ignore
from sklearn.model_selection import KFold # type: ignore
from scipy.stats import ttest_ind, shapiro, bartlett, levene, pearsonr, anderson, norm, zscore 
from statsmodels.robust.norms import HuberT, Hampel # 
from statsmodels.robust.robust_linear_model import RLM # 
from linearmodels.panel import PanelOLS 
import warnings 
import patsy 
import pingouin as pg 

from code.CENTRALraw import CENTRAL

df = pd.DataFrame(CENTRAL)

def process_CENTRAL():
    # Initialize DataFrame with provided data
    df = pd.DataFrame(CENTRAL)

# Step 8: Define functions to determine NBArev presence, wage tax presence, sales tax presence, and add NBA revenue flag

def set_group_presence(CENTRAL):
    NBArev_cities = ['PHL', 'CLE', 'DET', 'MIA', 'MEM', 'DAL', 'HOU', 'SAC', 'MKE', 'TOR', 'DEN']
    wagetax_cities = ['PHL', 'CLE', 'DET', 'STL', 'PIT', 'CIN', 'BAL']
    salestax_cities = ['PHL', 'DET', 'MIA', 'SAC', 'HOU', 'SPO', 'TAC', 'DEN', 'BTR']

    # Set flags for NBA revenue presence, wage tax, and sales tax
    for city_code, city_info in CENTRAL['Cities'].items():
        city_info['has_NBArev'] = 1 if city_code in NBArev_cities else 0
        city_info['has_wagetax'] = 1 if city_code in wagetax_cities else 0
        city_info['has_salestax'] = 1 if city_code in salestax_cities else 0

set_group_presence(CENTRAL)

# Step 9: Convert dictionary to DataFrame
CENTRAL_list = [
    {
        "Year": pd.to_datetime(year, format='%Y'),
        "City": city_code,
        "Population": city_info.get("pop", [None] * len(CENTRAL["YEARS"]))[min(CENTRAL["YEARS"].index(year), len(city_info.get("pop", []))-1)],
        "wagetax": (city_info.get("wagetax") if isinstance(city_info.get("wagetax"), list) else [None] * len(CENTRAL["YEARS"]))[min(CENTRAL["YEARS"].index(year), len(city_info.get("wagetax", []))-1)],
        "salestax": (city_info.get("salestax") if isinstance(city_info.get("salestax"), list) else [None] * len(CENTRAL["YEARS"]))[min(CENTRAL["YEARS"].index(year), len(city_info.get("salestax", []))-1)],
        "NBArev": (city_info.get("NBArev") if isinstance(city_info.get("NBArev"), list) else [None] * len(CENTRAL["YEARS"]))[min(CENTRAL["YEARS"].index(year), len(city_info.get("NBArev", []))-1)],
        "pay": city_info.get("pay", [None] * len(CENTRAL["YEARS"]))[min(CENTRAL["YEARS"].index(year), len(city_info.get("pay", []))-1)],
        "CPI": CENTRAL["National"]["CPI"][min(CENTRAL["YEARS"].index(year), len(CENTRAL["National"]["CPI"])-1)],
        "GDP": CENTRAL["National"]["GDP"][min(CENTRAL["YEARS"].index(year), len(CENTRAL["National"]["GDP"])-1)],
        "PCE": CENTRAL["National"]["PCE"][min(CENTRAL["YEARS"].index(year), len(CENTRAL["National"]["PCE"])-1)],
        "has_NBArev": city_info['has_NBArev'],
        "has_wagetax": city_info['has_wagetax'],
        "has_salestax": city_info['has_salestax'],
        "Intervention_2016": 1 if year == 2016 else 0,
        "post_2016": 1 if year >= 2016 else 0
    }
    for year in CENTRAL["YEARS"] for city_code, city_info in CENTRAL["Cities"].items()
]

df = pd.DataFrame(CENTRAL_list)

# Step 10: Verify the correct assignment of flags
print(df[['City', 'has_NBArev', 'has_wagetax', 'has_salestax']].drop_duplicates())

# Step 11: Smoothing pandemic effects
def smooth_pandemic_effects(CENTRAL):
    indices = {year: CENTRAL['YEARS'].index(year) for year in [2020, 2021] if year in CENTRAL['YEARS']}
    for city_code, city_info in CENTRAL['Cities'].items():
        for key in ['wagetax', 'salestax', 'NBArev']:
            if key in city_info:
                values = city_info[key]
                for year, idx in indices.items():
                    if len(values) > idx + 1 and values[idx - 1] is not None and values[idx + 1] is not None:
                        values[idx] = (values[idx - 1] + values[idx + 1]) / 2
                    elif idx > 0 and values[idx - 1] is not None:
                        values[idx] = values[idx - 1]
                    elif len(values) > idx + 1 and values[idx + 1] is not None:
                        values[idx] = values[idx + 1]

smooth_pandemic_effects(CENTRAL)

# Step 12: Adjust for inflation
def adjust_for_inflation(CENTRAL):
    cpi_index = {year: cpi for year, cpi in zip(CENTRAL['YEARS'], CENTRAL['National']['CPI'])}
    for city_code, city_info in CENTRAL['Cities'].items():
        for key in ['wagetax', 'salestax', 'NBArev']:
            if key in city_info:
                original_values = city_info[key]
                adjusted_values = []
                for year, value in zip(data['YEARS'], original_values):
                    if value is not None and year in cpi_index:
                        adjusted_value = value / (cpi_index[year] / 100)
                        adjusted_values.append(adjusted_value)
                    else:
                        adjusted_values.append(value)
                city_info[key] = adjusted_values

adjust_for_inflation(CENTRAL)

# Step 13: adjust for population
def adjust_for_per_capita(CENTRAL):
    for city_code, city_info in CENTRAL['Cities'].items():
        population = city_info.get('pop', [])
        for key in ['wagetax', 'salestax', 'NBArev']:
            if key in city_info:
                original_values = city_info[key]
                per_capita_values = []
                for pop, value in zip(population, original_values):
                    if value is not None and pop is not None:
                        per_capita_value = value / pop
                        per_capita_values.append(per_capita_value)
                    else:
                        per_capita_values.append(value)
                city_info[key] = per_capita_values

adjust_for_per_capita(CENTRAL)

# Step 14: Adjust for population
def adjust_for_per_capita(CENTRAL):
    for city_code, city_info in CENTRAL['Cities'].items():
        population = city_info.get('pop', [])
        for key in ['wagetax', 'salestax', 'NBArev']:
            if key in city_info:
                original_values = city_info[key]
                per_capita_values = []
                for pop, value in zip(population, original_values):
                    if value is not None and pop is not None:
                        per_capita_value = value / pop
                        per_capita_values.append(per_capita_value)
                    else:
                        per_capita_values.append(value)
                city_info[key] = per_capita_values

adjust_for_per_capita(CENTRAL)

# Step 15: Graphically show intervention with aggregated lines for NBA revenue, sales tax, and wage tax
# Extract years
years = CENTRAL['YEARS']

# Cities to include in the aggregation
cities_to_aggregate = ['PHL', 'CLE', 'DET', 'MIA', 'MEM', 'DAL', 'HOU', 'SAC', 'MKE', 'TOR', 'DEN']

# Initialize lists to store aggregated values
nba_rev_agg = []
sales_tax_agg = []
wage_tax_agg = []

# Calculate aggregated values for each year
for i in range(len(years)):
    nba_rev_total = sum(CENTRAL['Cities'][city]['NBArev'][i] for city in cities_to_aggregate if CENTRAL['Cities'][city]['NBArev'][i] is not None)
    sales_tax_total = sum(CENTRAL['Cities'][city]['salestax'][i] for city in cities_to_aggregate if CENTRAL['Cities'][city]['salestax'][i] is not None)
    wage_tax_total = sum(CENTRAL['Cities'][city]['wagetax'][i] for city in cities_to_aggregate if CENTRAL['Cities'][city]['wagetax'][i] is not None)
    
    nba_rev_agg.append(nba_rev_total)
    sales_tax_agg.append(sales_tax_total)
    wage_tax_agg.append(wage_tax_total)

# Create a plot
plt.figure(figsize=(14, 8))

# Plot aggregated NBA revenue
plt.plot(years, nba_rev_agg, marker='o', label='Aggregated NBA Revenue')

# Plot aggregated sales tax
plt.plot(years, sales_tax_agg, marker='s', label='Aggregated Sales Tax')

# Plot aggregated wage tax
plt.plot(years, wage_tax_agg, marker='^', label='Aggregated Wage Tax')

# Highlight the intervention year 2016
plt.axvline(x=2016, color='r', linestyle='--', label='2016 Media Deal')

# Adding title and labels
plt.title('Aggregated NBA Revenue, Sales Tax, and Wage Tax for Selected Cities (2001-2022)')
plt.xlabel('Year')
plt.ylabel('Amount (in millions)')
plt.legend()
plt.grid(True)
plt.show()

# Step 16: Parallel Trends for Sales Tax

# Years for analysis
years_analysis = [year for year in years if 2001 <= year <= 2014]

# Cities with and without NBA team and a sales tax
cities_with_nba_sales_tax = ['PHL', 'DET', 'MIA', 'SAC', 'HOU', 'DEN']
cities_without_nba_sales_tax = ['SPO', 'TAC', 'BTR']

# Initialize lists for aggregated values
sales_tax_nba, sales_tax_non_nba = [], []

# Aggregate values for each year
for year in years_analysis:
    # Sales Tax
    sales_tax_nba_total = sum(
        CENTRAL['Cities'][city]['salestax'][years.index(year)]
        for city in cities_with_nba_sales_tax if CENTRAL['Cities'][city]['salestax'][years.index(year)] is not None
    )
    sales_tax_non_nba_total = sum(
        CENTRAL['Cities'][city]['salestax'][years.index(year)]
        for city in cities_without_nba_sales_tax if CENTRAL['Cities'][city]['salestax'][years.index(year)] is not None
    )
    sales_tax_nba.append(sales_tax_nba_total)
    sales_tax_non_nba.append(sales_tax_non_nba_total)

# Create a plot for sales tax trends
plt.figure(figsize=(14, 8))
plt.subplot(2, 1, 1)
plt.plot(years_analysis, sales_tax_nba, marker='o', label='Sales Tax - NBA Cities')
plt.plot(years_analysis, sales_tax_non_nba, marker='s', label='Sales Tax - Non-NBA Cities')

# Fit linear regression models to the sales tax data
X_years_analysis = np.array(years_analysis).reshape(-1, 1)
model_nba_sales = LinearRegression().fit(X_years_analysis, sales_tax_nba)
model_non_nba_sales = LinearRegression().fit(X_years_analysis, sales_tax_non_nba)

# Predict sales tax using the fitted models
sales_tax_nba_pred = model_nba_sales.predict(X_years_analysis)
sales_tax_non_nba_pred = model_non_nba_sales.predict(X_years_analysis)

# Plot the regression lines
plt.plot(years_analysis, sales_tax_nba_pred, linestyle='--', color='blue', label='Trend Line - NBA Cities')
plt.plot(years_analysis, sales_tax_non_nba_pred, linestyle='--', color='orange', label='Trend Line - Non-NBA Cities')
plt.title('Parallel Trends: Sales Tax in Cities With and Without NBA Teams (2001-2014)')
plt.xlabel('Year')
plt.ylabel('Sales Tax (in millions)')
plt.legend()
plt.grid(True)

# Calculate and print the slopes
slope_sales_nba = model_nba_sales.coef_[0]
slope_sales_non_nba = model_non_nba_sales.coef_[0]
print(f"Slope of sales tax trend for NBA cities: {slope_sales_nba}")
print(f"Slope of sales tax trend for non-NBA cities: {slope_sales_non_nba}")

# Statistical test for parallel trends (Sales Tax)
ttest_sales_result = ttest_ind(sales_tax_nba_pred, sales_tax_non_nba_pred)
print(f"T-test result for parallel trends (Sales Tax): {ttest_sales_result}")

# Perform statistical validation for parallel trends using the slopes (Sales Tax)
slope_sales_diff = slope_sales_nba - slope_sales_non_nba
slope_sales_diff_std = np.std(sales_tax_nba_pred - sales_tax_non_nba_pred) / np.sqrt(len(years_analysis))
t_statistic_sales = slope_sales_diff / slope_sales_diff_std
p_value_sales = 2 * (1 - norm.cdf(abs(t_statistic_sales)))
print(f"Slope difference (Sales Tax): {slope_sales_diff}")
print(f"T-statistic (Sales Tax): {t_statistic_sales}")
print(f"P-value (Sales Tax): {p_value_sales}")

# Step 16.5: Parallel Trends for Wage Tax

# Cities with a wage tax and an NBA team
cities_with_nba_wage_tax = ['PHL', 'CLE', 'DET']

# Cities with a wage tax and no NBA team
cities_with_wage_no_nba = ['STL', 'PIT', 'CIN', 'BAL']

# Initialize lists for aggregated values for wage tax
wage_tax_with_nba, wage_tax_without_nba = [], []

# Aggregate values for each year
for year in years_analysis:
    wage_tax_with_nba_total = sum(
        CENTRAL['Cities'][city]['wagetax'][years.index(year)]
        for city in cities_with_nba_wage_tax if CENTRAL['Cities'][city]['wagetax'][years.index(year)] is not None
    )
    wage_tax_without_nba_total = sum(
        CENTRAL['Cities'][city]['wagetax'][years.index(year)]
        for city in cities_with_wage_no_nba if CENTRAL['Cities'][city]['wagetax'][years.index(year)] is not None
    )
    wage_tax_with_nba.append(wage_tax_with_nba_total)
    wage_tax_without_nba.append(wage_tax_without_nba_total)

# Create a plot for wage tax trends
plt.subplot(2, 1, 2)
plt.plot(years_analysis, wage_tax_with_nba, marker='o', label='Wage Tax - Cities with NBA Team')
plt.plot(years_analysis, wage_tax_without_nba, marker='s', label='Wage Tax - Cities without NBA Team')

# Fit linear regression models to the wage tax data
model_with_nba_wage = LinearRegression().fit(X_years_analysis, wage_tax_with_nba)
model_without_nba_wage = LinearRegression().fit(X_years_analysis, wage_tax_without_nba)

# Predict wage tax using the fitted models
wage_tax_with_nba_pred = model_with_nba_wage.predict(X_years_analysis)
wage_tax_without_nba_pred = model_without_nba_wage.predict(X_years_analysis)

# Plot the regression lines
plt.plot(years_analysis, wage_tax_with_nba_pred, linestyle='--', color='blue', label='Trend Line - Cities with NBA Team')
plt.plot(years_analysis, wage_tax_without_nba_pred, linestyle='--', color='orange', label='Trend Line - Cities without NBA Team')
plt.title('Parallel Trends: Wage Tax in Cities With and Without NBA Teams (2001-2014)')
plt.xlabel('Year')
plt.ylabel('Wage Tax (in millions)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Calculate and print the slopes
slope_wage_with_nba = model_with_nba_wage.coef_[0]
slope_wage_without_nba = model_without_nba_wage.coef_[0]
print(f"Slope of wage tax trend for cities with NBA team: {slope_wage_with_nba}")
print(f"Slope of wage tax trend for cities without NBA team: {slope_wage_without_nba}")

# Statistical test for parallel trends (Wage Tax)
ttest_wage_result = ttest_ind(wage_tax_with_nba_pred, wage_tax_without_nba_pred)
print(f"T-test result for parallel trends (Wage Tax): {ttest_wage_result}")

# Perform statistical validation for parallel trends using the slopes (Wage Tax)
slope_wage_diff = slope_wage_with_nba - slope_wage_without_nba
slope_wage_diff_std = np.std(wage_tax_with_nba_pred - wage_tax_without_nba_pred) / np.sqrt(len(years_analysis))
t_statistic_wage = slope_wage_diff / slope_wage_diff_std
p_value_wage = 2 * (1 - norm.cdf(abs(t_statistic_wage)))
print(f"Slope difference (Wage Tax): {slope_wage_diff}")
print(f"T-statistic (Wage Tax): {t_statistic_wage}")
print(f"P-value (Wage Tax): {p_value_wage}")

#17 multivariate did w WLS

# Prepare the data for the combined sales and wage tax DiD analysis (2016 intervention year)
years_analysis = [year for year in years if year <= 2016]
CENTRAL = []

for year in years_analysis:
    for city_code, city_info in CENTRAL['Cities'].items():
        nba_team = 1 if city_info['has_NBArev'] else 0
        salestax = city_info['salestax'][years.index(year)] if 'salestax' in city_info else None
        wagetax = city_info['wagetax'][years.index(year)] if 'wagetax' in city_info else None
        post_2016 = 1 if year >= 2016 else 0
        
        if salestax is not None or wagetax is not None:
            CENTRAL_list_combined.append({
                'Year': year,
                'City': city_code,
                'SalesTax': salestax,
                'WageTax': wagetax,
                'NBA_Team': nba_team,
                'Post_2016': post_2016,
                'Interaction_2016': nba_team * post_2016
            })

combined_tax_did_df = pd.DataFrame(CENTRAL_list_combined)

# Drop rows where both SalesTax and WageTax are NaN
combined_tax_did_df = combined_tax_did_df.dropna(subset=['SalesTax', 'WageTax'], how='all')

# Separate data for SalesTax and WageTax for WLS modeling
combined_tax_did_df_sales = combined_tax_did_df.dropna(subset=['SalesTax'])
combined_tax_did_df_wage = combined_tax_did_df.dropna(subset=['WageTax'])

# WLS model for SalesTax
X_sales = combined_tax_did_df_sales[['NBA_Team', 'Post_2016', 'Interaction_2016']]
X_sales = sm.add_constant(X_sales)
y_sales = combined_tax_did_df_sales['SalesTax']

weights_sales = 1 / combined_tax_did_df_sales['SalesTax'].var()
sales_tax_model_wls = sm.WLS(y_sales, X_sales, weights=weights_sales).fit(cov_type='HC3')

print("Sales Tax Model Summary with WLS and 2016 Intervention Year")
print(sales_tax_model_wls.summary())

# WLS model for WageTax
X_wage = combined_tax_did_df_wage[['NBA_Team', 'Post_2016', 'Interaction_2016']]
X_wage = sm.add_constant(X_wage)
y_wage = combined_tax_did_df_wage['WageTax']

weights_wage = 1 / combined_tax_did_df_wage['WageTax'].var()
wage_tax_model_wls = sm.WLS(y_wage, X_wage, weights=weights_wage).fit(cov_type='HC3')

print("Wage Tax Model Summary with WLS and 2016 Intervention Year")
print(wage_tax_model_wls.summary())

# Step 18: Diagnostics. Multicollinearity, Normality, and Heteroscedasticity Check for Both Models

# Multicollinearity Test (VIF) for sales tax model
print("Variance Inflation Factor (VIF) for Sales Tax Model:")
vif_data_sales = pd.DataFrame()
vif_data_sales["Feature"] = X_sales.columns
vif_data_sales["VIF"] = [variance_inflation_factor(X_sales.values, i) for i in range(X_sales.shape[1])]
print(vif_data_sales)
print("\n")

# Normality Test (Shapiro-Wilk) for sales tax model
residuals_sales_tax = sales_tax_model_wls.resid
shapiro_test_sales_tax = shapiro(residuals_sales_tax)
print(f"Shapiro-Wilk test for normality (Sales Tax): p-value = {shapiro_test_sales_tax[1]}")
print("\n")

# Heteroscedasticity Test (Breusch-Pagan) for sales tax model
bp_test_sales_tax = het_breuschpagan(residuals_sales_tax, X_sales)
print(f"Breusch-Pagan test for heteroscedasticity (Sales Tax): p-value = {bp_test_sales_tax[1]}")
print("\n")

# Multicollinearity Test (VIF) for wage tax model
print("Variance Inflation Factor (VIF) for Wage Tax Model:")
vif_data_wage = pd.DataFrame()
vif_data_wage["Feature"] = X_wage.columns
vif_data_wage["VIF"] = [variance_inflation_factor(X_wage.values, i) for i in range(X_wage.shape[1])]
print(vif_data_wage)
print("\n")

# Normality Test (Shapiro-Wilk) for wage tax model
residuals_wage_tax = wage_tax_model_wls.resid
shapiro_test_wage_tax = shapiro(residuals_wage_tax)
print(f"Shapiro-Wilk test for normality (Wage Tax): p-value = {shapiro_test_wage_tax[1]}")
print("\n")

# Heteroscedasticity Test (Breusch-Pagan) for wage tax model
bp_test_wage_tax = het_breuschpagan(residuals_wage_tax, X_wage)
print(f"Breusch-Pagan test for heteroscedasticity (Wage Tax): p-value = {bp_test_wage_tax[1]}")

# Step 19: Stationary Bootstraping
import numpy as np
from scipy.stats import norm

# Assuming residuals_sales_tax and residuals_wage_tax are already defined
np.random.seed(42)  # For reproducibility
n_iterations = 10000

# Convert residuals to numpy arrays
residuals_sales_tax = np.array(residuals_sales_tax)
residuals_wage_tax = np.array(residuals_wage_tax)

# Generate bootstrap samples for Sales Tax Residuals
bootstrap_samples_sales = np.zeros((n_iterations, len(residuals_sales_tax)))
for i in range(n_iterations):
    indices = np.random.randint(0, len(residuals_sales_tax), len(residuals_sales_tax))
    bootstrap_samples_sales[i] = residuals_sales_tax[indices]

# Generate bootstrap samples for Wage Tax Residuals
bootstrap_samples_wage = np.zeros((n_iterations, len(residuals_wage_tax)))
for i in range(n_iterations):
    indices = np.random.randint(0, len(residuals_wage_tax), len(residuals_wage_tax))
    bootstrap_samples_wage[i] = residuals_wage_tax[indices]

# Check the shape of the bootstrap samples to confirm
print("Bootstrap samples for Sales Tax Residuals shape:", bootstrap_samples_sales.shape)
print("Bootstrap samples for Wage Tax Residuals shape:", bootstrap_samples_wage.shape)

# Calculate percentiles for 95% confidence intervals
alpha = 0.05
lower_percentile = 100 * (alpha / 2)
upper_percentile = 100 * (1 - alpha / 2)

# Calculate confidence intervals for Sales Tax Residuals
lower_bound_sales = np.percentile(bootstrap_samples_sales, lower_percentile, axis=0)
upper_bound_sales = np.percentile(bootstrap_samples_sales, upper_percentile, axis=0)

# Calculate confidence intervals for Wage Tax Residuals
lower_bound_wage = np.percentile(bootstrap_samples_wage, lower_percentile, axis=0)
upper_bound_wage = np.percentile(bootstrap_samples_wage, upper_percentile, axis=0)

# Print confidence intervals
print("95% Confidence Interval for Sales Tax Residuals:")
print("Lower Bound:\n", lower_bound_sales)
print("Upper Bound:\n", upper_bound_sales)

print("\n95% Confidence Interval for Wage Tax Residuals:")
print("Lower Bound:\n", lower_bound_wage)
print("Upper Bound:\n", upper_bound_wage)

# Conduct hypothesis testing for Sales Tax Residuals
mean_sales = np.mean(bootstrap_samples_sales, axis=0)
z_scores_sales = (mean_sales - np.mean(residuals_sales_tax)) / np.std(bootstrap_samples_sales, axis=0)
p_values_sales = 2 * (1 - norm.cdf(np.abs(z_scores_sales)))

# Conduct hypothesis testing for Wage Tax Residuals
mean_wage = np.mean(bootstrap_samples_wage, axis=0)
z_scores_wage = (mean_wage - np.mean(residuals_wage_tax)) / np.std(bootstrap_samples_wage, axis=0)
p_values_wage = 2 * (1 - norm.cdf(np.abs(z_scores_wage)))

# Print p-values
print("\nP-values for hypothesis testing of Sales Tax Residuals:\n", p_values_sales)
print("\nP-values for hypothesis testing of Wage Tax Residuals:\n", p_values_wage)

return df