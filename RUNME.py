import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
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

# PHILADELPHIA EXPLORATORY ANALYSIS #################################

# Step 1: Initialize DataFrame with provided data
df = pd.DataFrame({
    "Year": [i.year for i in pd.date_range(start="2001", periods=22, freq='YS')],
    "76ers_Revenue": [106000000, 109000000, 107000000, 110000000, 110000000, 112000000, 116000000, 115000000, 110000000, 116000000, 107000000, 117000000, 125000000, 124000000, 140000000, 184000000, 268000000, 300000000, 259000000, 236000000, 345000000, 371000000],
    "76ers_Operating_Income": [18100000, 2000000, 700000, 7200000, -6200000, -2800000, 300000, 7600000, -1200000, -10300000, -800000, -3800000, 24400000, 13900000, 18200000, 40000000, 68000000, 90000000, 51000000, 13000000, 87000000, 120000000],
    "76ers_Player_Expenses": [53000000, 61000000, 63000000, 70000000, 78000000, 75000000, 78000000, 70000000, 69000000, 73000000, 60000000, 83000000, 60000000, 60000000, 74000000, 89000000, 105000000, 117000000, 120000000, 128000000, 143000000, 161000000],
    "76ers_Gate_Receipts": [21680000, 22120000, 22570000, 23030000, 23500000, 23980000, 24470000, 24970000, 25480000, 26000000, 26000000, 27000000, 22000000, 21000000, 23000000, 26000000, 59000000, 71000000, 48000000, 13000000, 83000000, 122000000],
    "76ers_PILOT": [2000000] * 22,
    "Amusement_Tax_Rate": [0.05] * 22,
    "Net_Profit_Tax_Rate_Resident": [0.045385, 0.045, 0.044625, 0.04431, 0.04398, 0.04329, 0.0426, 0.04219, 0.04182, 0.0398, 0.0393, 0.039296, 0.03928, 0.03924, 0.0392, 0.03916, 0.0388, 0.0384, 0.038, 0.0376, 0.0372, 0.0368],
    "BIRT_Gross_Receipts_Rate": [0.002525, 0.0024, 0.0023, 0.0021, 0.0019, 0.001665, 0.00154, 0.001415, 0.001415, 0.001415, 0.001415, 0.001415, 0.001415, 0.001415, 0.001415, 0.001415, 0.001415, 0.001415, 0.001415, 0.001415, 0.001415, 0.001415],
    "BIRT_Net_Income_Rate": [0.065, 0.065, 0.065, 0.065, 0.065, 0.065, 0.065, 0.0645, 0.0643, 0.0641, 0.0639, 0.0635, 0.063, 0.0625, 0.0625, 0.062, 0.0599, 0.0599, 0.0599, 0.062, 0.0599, 0.0599],
    "Wage_Tax_Rate_Resident": [0.045385, 0.045, 0.044625, 0.04431, 0.04398, 0.04329, 0.0426, 0.04219, 0.04182, 0.0398, 0.0393, 0.039296, 0.03928, 0.03924, 0.0392, 0.03916, 0.0388, 0.0384, 0.038, 0.0376, 0.0372, 0.0368],
    "Wage_Tax_Rate_Non_Resident": [0.043125, 0.043125, 0.043125, 0.043125, 0.043125, 0.043125, 0.043125, 0.043125, 0.043125, 0.043125, 0.043125, 0.043125, 0.043125, 0.043125, 0.043125, 0.043125, 0.043125, 0.043125, 0.043125, 0.043125, 0.043125, 0.043125],
    "Sales_Tax_Rate": [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
    "PHL_Business_Priv": [314000000, 295800000, 286100000, 309200000, 379500000, 415500000, 436400000, 398800000, 386000000, 364700000, 399200000, 452400000, 469200000, 453400000, 505600000, 440200000, 456100000, 446071000, 556100000, 534239000, 541598000, 749865000],
    "PHL_Wage_and_Earnings": [1059000000, 1019000000, 1025100000, 1063600000, 1087300000, 1125800000, 1182700000, 1197300000, 1129200000, 1128700000, 1127400000, 1192200000, 1255900000, 1318800000, 1325064000, 1364612000, 1448900000, 1536868000, 1581900000, 1599162000, 1450745000, 1653878000],
    "PHL_Sales_and_Use": [111300000, 108100000, 108000000, 108000000, 119900000, 127800000, 132600000, 137300000, 128200000, 207100000, 244600000, 253500000, 257600000, 263100000, 149500000, 169383000, 188400000, 198400000, 224200000, 204591000, 209740000, 277690000]
})

# Step 1.5: Use original tax rates without inflation adjustments
df['76ers_BIRT_Gross_Receipts_Rate'] = df['BIRT_Gross_Receipts_Rate']
df['76ers_BIRT_Net_Income_Rate'] = df['BIRT_Net_Income_Rate']
df['76ers_Wage_Tax_Rate_Resident'] = df['Wage_Tax_Rate_Resident']
df['76ers_Wage_Tax_Rate_Non_Resident'] = df['Wage_Tax_Rate_Non_Resident']
df['76ers_Sales_Tax_Rate'] = df['Sales_Tax_Rate']
df['76ers_Net_Profit_Tax_Rate_Resident'] = df['Net_Profit_Tax_Rate_Resident']

# Step 2: Calculate tax revenue contributions using the original tax rates
df['76ers_Amusement_Tax_Contribution'] = df['76ers_Gate_Receipts'] * df['Amusement_Tax_Rate']
df['76ers_BIRT_Gross_Receipts_Contribution'] = df['76ers_Revenue'] * df['76ers_BIRT_Gross_Receipts_Rate']
df['76ers_BIRT_Net_Income_Contribution'] = df['76ers_Operating_Income'] * df['76ers_BIRT_Net_Income_Rate']
df['76ers_Wage_Tax_Contribution_Resident'] = df['76ers_Player_Expenses'] * (1/3) * df['76ers_Wage_Tax_Rate_Resident']
df['76ers_Wage_Tax_Contribution_NonResident'] = df['76ers_Player_Expenses'] * (2/3) * df['76ers_Wage_Tax_Rate_Non_Resident']
df['76ers_Sales_Tax_Contribution'] = df['76ers_Revenue'] * df['76ers_Sales_Tax_Rate']
df['76ers_Net_Profits_Tax_Contribution'] = df['76ers_Operating_Income'] * df['76ers_Net_Profit_Tax_Rate_Resident']

# Step 2.5: Combine BIRT Contributions and Wage Tax Contributions for 76ers
df['76ers_Total_BIRT_Contribution'] = df['76ers_BIRT_Gross_Receipts_Contribution'] + df['76ers_BIRT_Net_Income_Contribution']
df['76ers_Total_Wage_Tax_Contribution'] = df['76ers_Wage_Tax_Contribution_Resident'] + df['76ers_Wage_Tax_Contribution_NonResident']

# Step 3: Calculate total tax revenue per year from 76ers
df['76ers_Total_Revenue_Per_Year'] = df[['76ers_Amusement_Tax_Contribution', '76ers_BIRT_Gross_Receipts_Contribution', '76ers_BIRT_Net_Income_Contribution', '76ers_Wage_Tax_Contribution_Resident', '76ers_Wage_Tax_Contribution_NonResident', '76ers_Sales_Tax_Contribution', '76ers_Net_Profits_Tax_Contribution', '76ers_PILOT']].sum(axis=1)

# Step 4: Clip negative values to 0
tax_contribution_columns = [
    '76ers_Amusement_Tax_Contribution',
    '76ers_BIRT_Gross_Receipts_Contribution',
    '76ers_BIRT_Net_Income_Contribution',
    '76ers_Wage_Tax_Contribution_Resident',
    '76ers_Wage_Tax_Contribution_NonResident',
    '76ers_Sales_Tax_Contribution',
    '76ers_Net_Profits_Tax_Contribution',
    '76ers_PILOT'
]
df[tax_contribution_columns] = df[tax_contribution_columns].clip(lower=0)

# Step 4.5: Print yearly tax contributions from the 76ers
print("Yearly Tax Contributions from the 76ers (No Negative Contributions):\n")
headers = ["Year", "Amusement Tax", "BIRT Gross Receipts", "BIRT Net Income",
           "Wage Tax Resident", "Wage Tax Non-Resident", "Sales Tax",
           "Net Profits Tax", "PILOT", "Total Revenue"]
header_format = "{:<6} " + "{:<20} " * (len(headers) - 1)
row_format = "{:<6} $" + "${:<18,d} " * 8

print(header_format.format(*headers))

for index, row in df.iterrows():
    print(row_format.format(
        row['Year'],
        int(row['76ers_Amusement_Tax_Contribution']),
        int(row['76ers_BIRT_Gross_Receipts_Contribution']),
        int(row['76ers_BIRT_Net_Income_Contribution']),
        int(row['76ers_Wage_Tax_Contribution_Resident']),
        int(row['76ers_Wage_Tax_Contribution_NonResident']),
        int(row['76ers_Sales_Tax_Contribution']),
        int(row['76ers_Net_Profits_Tax_Contribution']),
        int(row['76ers_PILOT']),
        int(row['76ers_Total_Revenue_Per_Year'])
    ))

# Step 4.6: Print Philadelphia tax revenues by year
print("Philadelphia Tax Revenues By Year:\n")
headers = ["Year", "Business Privilege", "Wage and Earnings", "Sales and Use"]
header_format = "{:<6} " + "{:<20} " * (len(headers) - 1)
row_format = "{:<6} $" + "${:<18,d} " * 3

print(header_format.format(*headers))

for index, row in df.iterrows():
    print(row_format.format(
        row['Year'],
        int(row['PHL_Business_Priv']),
        int(row['PHL_Wage_and_Earnings']),
        int(row['PHL_Sales_and_Use'])
    ))



# Multi City Study ##############################################################

# Step 7: Initiate new multi-city dataframe

data = {
    "YEARS": [
        2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022
    
    ],

    "Cities": {
        "PHL": {
            "City": "Philadelphia",
            "pop": [1500165, 1488689, 1479323, 1473598, 1466715, 1459427, 1458905, 1459979, 1469192, 1528303, 1540615, 1552087, 1558713, 1565949, 1571679, 1576604, 1581699, 1586422, 1584439, 1601005, 1576251, 1567258],
            "wagetax": [1047000000, 1006000000, 1013000000, 1049000000, 1073000000, 1111000000, 1167000000, 1184000000, 1117000000, 1114000000, 1134000000, 1196000000, 1221000000, 1261000000, 1325000000, 1373000000, 1448000000, 1542000000, 1581000000, 1599000000, 1450000000, 1653000000],
            "pay": [53000000, 61000000, 63000000, 70000000, 78000000, 75000000, 78000000, 70000000, 69000000, 73000000, 60000000, 83000000, 60000000, 60000000, 74000000, 89000000, 105000000, 117000000, 120000000, 128000000, 143000000, 161000000],
            "salestax": [111300000, 108100000, 108000000, 108000000, 119900000, 127800000, 132600000, 137300000, 128000000, 207113000, 244600000, 253500000, 257600000, 263100000, 139458000, 169383000, 188400000, 198400000, 224200000, 204591000, 230408000, 277690000],
            "NBArev": [106000000, 109000000, 107000000, 110000000, 110000000, 112000000, 116000000, 115000000, 110000000, 116000000, 107000000, 117000000, 125000000, 124000000, 140000000, 184000000, 268000000, 300000000, 259000000, 236000000, 345000000, 371000000]
        },
 
        "CLE": {
            "City": "Cleveland",
            "pop": [472360, 467904, 463071, 457516, 451317, 444688, 440358, 436379, 433784, 396095, 392862, 391767, 392040, 391129, 389524, 387885, 385490, 383408, 381386, 372032, 367991, 361607],
            "wagetax": [19325093, 20205811, 19248400, 19081984, 17957565, 18087545, 19120031, 19397366, 17186144, 16442045, 17169309, 16650005, 17435006, 16923629, 18410061, 18965891, 19131883, 17378944, 17353046, 14631063, 16592567, 19268900],
            "pay": [46000000, 53000000, 50000000, 52000000, 55000000, 69000000, 85000000, 94000000, 90000000, 58000000, 57000000, 72000000, 74000000, 87000000, 131000000, 149000000, 152000000, 137000000, 120000000, 110000000, 128000000, 158000000],
            "salestax": [None] * 22,
            "NBArev": [79000000, 72000000, 93000000, 102000000, 115000000, 152000000, 159000000, 159000000, 161000000, 149000000, 128000000, 145000000, 149000000, 191000000, 233000000, 280000000, 302000000, 300000000, 269000000, 211000000, 325000000, 348000000]
        },
        "DET": {
            "City": "Detroit",
            "pop": [933204, 918927, 907000, 892644, 877791, 862850, 848438, 831626, 821019, 711198, 705202, 700289, 691986, 682746, 679562, 677379, 675031, 673658, 670442, 638176, 632464, 620376],
            "wagetax": [341000000, 323000000, 310000000, 290000000, 284000000, 278000000, 276000000, 240000000, 217000000, 216000000, 228000000, 233000000, 248000000, 253000000, 263000000, 263000000, 284000000, 310000000, 361000000, 290000000, 316000000, 402000000],
            "pay": [42000000, 51000000, 57000000, 58000000, 63000000, 63000000, 70000000, 75000000, 64000000, 70000000, 59000000, 76000000, 69000000, 71000000, 81000000, 123000000, 127000000, 138000000, 123000000, 100000000, 127000000, 132000000],
            "salestax": [185732000, 203321599, 174532811, 190928138, 183294833, 189253428, 243533764, 258599558, 237044188, 196333386, 201253031, 197066068, 138617705, 138253202, 131066385, 120618786, 133262994, 144595148, 166904971, 171238657, 156967911, 177707399],
            "NBArev": [95000000, 102000000, 121000000, 134000000, 138000000, 154000000, 160000000, 171000000, 147000000, 141000000, 125000000, 139000000, 144000000, 154000000, 172000000, 221000000, 235000000, 255000000, 227000000, 192000000, 278000000, 274000000]
        },
        "STL": {
            "City": "St. Louis",
            "pop": [342054, 336291, 331738, 326864, 319868, 314071, 310927, 309864, 309659, 319367, 319417, 319534, 318624, 317599, 316268, 312926, 308424, 303661, 300887, 300528, 293310, 286578],
            "wagetax": [122594000, 155760000, 153923000, 148081000, 158533000, 170934000, 169822000, 181812000, 181214000, 172450000, 178571000, 186068000, 189762000, 202022000, 203251000, 204013000, 233018000, 218338000, 230627000, 234602000, 239914000, 257249000],
            "pay": [None] * 22,
            "salestax": [None] * 22,
            "NBArev": [None] * 22
        },
        "PIT": {
            "City": "Pittsburgh",
            "pop": [330686, 327859, 325932, 321549, 317559, 314787, 313674, 313470, 313440, 305351, 306117, 306455, 307006, 306080, 304224, 304789, 301606, 300669, 300056, 302777, 300431, 302898],
            "wagetax": [46684000, 47642000, 45924000, 46439000, 48238000, 49815000, 65296000, 65296000, 67483000, 69674000, 74146000, 75228000, 84103000, 86860000, 88853000, 92241000, 96627000, 99874000, 109713000, 108300000, 114450000, 125294000],
            "pay": [None] * 22,
            "salestax": [None] * 22,
            "NBArev": [None] * 22
        },
        "CIN": {
            "City": "Cincinnati",
            "pop": [327472, 322644, 318728, 314729, 309978, 307439, 306309, 305039, 305662, 297061, 296292, 297031, 298058, 298809, 299649, 299978, 301852, 302501, 304445, 310113, 308935, 309513],
            "wagetax": [188598000, 186259000, 187993000, 193411000, 210536000, 219000000, 225008000, 231758000, 223800000, 222497000, 233752000, 238210000, 227873000, 251683000, 261848000, 277673000, 274312000, 277477000, 288826000, 297701000, 330863000, 340427000],
            "pay": [None] * 22,
            "salestax": [None] * 22,
            "NBArev": [None] * 22
        },
        "BAL": {
            "City": "Baltimore",
            "pop": [638700, 630367, 623567, 614564, 610068, 607864, 606006, 603758, 601984, 620942, 620493, 623035, 622591, 623833, 622831, 616542, 610853, 603241, 594601, 583132, 576498, 569931],
            "wagetax": [170907000, 181574000, 173466000, 182506000, 199635000, 225517000, 243611000, 267625000, 262901000, 251731000, 234955000, 257893000, 276111000, 284437000, 300014000, 346727000, 335923000, 346797000, 440144000, 396540000, 410712000, 491092000],
            "pay": [None] * 22,
            "salestax": [None] * 22,
            "NBArev": [None] * 22
        },
        "MIA": {
            "City": "Miami",
            "pop": [365321, 369385, 372440, 374137, 382335, 397630, 406171, 412115, 416577, 400798, 406607, 411149, 415666, 425097, 434703, 448789, 455925, 461962, 467103, 441889, 439890, 449514],
            "wagetax": [None] * 22,
            "pay": [53000000, 49000000, 50000000, 63000000, 64000000, 68000000, 79000000, 74000000, 78000000, 69000000, 68000000, 90000000, 94000000, 89000000, 104000000, 104000000, 131000000, 144000000, 126000000, 112000000, 138000000, 161000000],
            "salestax": [99509000, 101523000, 111386000, 113947000, 118751000, 130538000, 130822000, 134017000, 113916000, 111092000, 123264000, 131392000, 140449000, 148654000, 157047000, 162740000, 163323000, 174312000, 176298000, 152278000, 189746000, 229932000],
            "NBArev": [96000000, 91000000, 93000000, 119000000, 132000000, 131000000, 131000000, 126000000, 124000000, 158000000, 150000000, 188000000, 188000000, 180000000, 210000000, 253000000, 259000000, 294000000, 266000000, 218000000, 326000000, 371000000]
        },
        "MEM": {
            "City": "Memphis",
            "pop": [685198, 683492, 682206, 680631, 679437, 680702, 678052, 675278, 675183, 652349, 655434, 658961, 657386, 655299, 654469, 653026, 651398, 651700, 650998, 631326, 628127, 621056],
            "wagetax": [None] * 22,
            "pay": [45000000, 54000000, 61000000, 70000000, 71000000, 66000000, 58000000, 59000000, 59000000, 75000000, 63000000, 71000000, 79000000, 81000000, 95000000, 125000000, 121000000, 136000000, 118000000, 106000000, 93000000, 132000000],
            "salestax": [None] * 22,
            "NBArev": [66000000, 63000000, 75000000, 98000000, 101000000, 98000000, 95000000, 88000000, 92000000, 99000000, 96000000, 126000000, 135000000, 147000000, 155000000, 206000000, 213000000, 224000000, 210000000, 186000000, 272000000, 258000000]
        },
        "DAL": {
            "City": "Dallas",
            "pop": [1202893, 1203699, 1202932, 1205260, 1211806, 1225737, 1236350, 1252932, 1280878, 1200360, 1218214, 1241950, 1258580, 1278737, 1300859, 1323587, 1342275, 1341692, 1343192, 1303234, 1288457, 1299544],
            "wagetax": [None] * 22,
            "pay": [58000000, 75000000, 83000000, 95000000, 97000000, 91000000, 103000000, 99000000, 81000000, 88000000, 67000000, 75000000, 82000000, 90000000, 86000000, 127000000, 92000000, 108000000, 112000000, 87000000, 117000000, 177000000],
            "salestax": [210749000, 194133000, 183229000, 194989000, 198441000, 217836000, 224078000, 
            231000000, 208000000, 206000000, 217000000, 231000000, 244000000, 257000000, 
            275000000, 286000000, 295000000, 307000000, 320000000, 314000000, 354000000, 
            407000000],
            "NBArev": [105000000, 117000000, 117000000, 124000000, 140000000, 140000000, 153000000, 154000000, 146000000, 166000000, 137000000, 162000000, 168000000, 177000000, 194000000, 233000000, 287000000, 307000000, 295000000, 231000000, 364000000, 429000000]
        },
        "HOU": {
            "City": "Houston",
            "pop": [1994315, 2015621, 2019727, 2017358, 2021875, 2058724, 2065128, 2084441, 2118595, 2097647, 2123298, 2158700, 2196367, 2238653, 2283616, 2306360, 2313079, 2314478, 2315720, 2300027, 2288250, 2302878],
            "wagetax": [None] * 22,
            "pay": [49000000, 53000000, 59000000, 63000000, 73000000, 68000000, 75000000, 78000000, 67000000, 70000000, 52000000, 68000000, 76000000, 85000000, 103000000, 108000000, 130000000, 146000000, 124000000, 113000000, 128000000, 145000000],
            "salestax": [329705000, 341952000, 322538000, 347982000, 370583000, 422598000, 461467000, 495173000, 507103000, 468965000, 492824000, 546543000, 600256000, 629441000, 667061000, 640476000, 631993000, 674279000, 692271000, 684425000, 706829000, 820622000],
            "NBArev": [82000000, 82000000, 125000000, 141000000, 142000000, 149000000, 156000000, 160000000, 153000000, 150000000, 135000000, 191000000, 175000000, 237000000, 244000000, 296000000, 326000000, 348000000, 308000000, 223000000, 350000000, 381000000]
        },
        "SAC": {
            "City": "Sacramento",
            "pop": [420930, 433222, 440951, 447466, 450381, 451399, 457477, 462712, 468051, 467246, 470788, 473430, 476645, 481284, 486741, 492284, 498386, 504185, 509748, 523416, 525041, 528001],
            "wagetax": [None] * 22,
            "pay": [55000000, 73000000, 72000000, 64000000, 67000000, 68000000, 67000000, 72000000, 72000000, 50000000, 45000000, 61000000, 69000000, 81000000, 84000000, 109000000, 103000000, 107000000, 100000000, 79000000, 141000000, 145000000],
            "salestax": [62588000, 59515000, 62018000, 66234000, 70627000, 72479000, 56441000, 54821000, 48905000, 45670000, 57121000, 47680000, 50683000, 57121000, 99615000, 102596000, 110212000, 125560000, 157816000, 194868000, 216170000, 248515000],
            "NBArev": [102000000, 102000000, 118000000, 119000000, 126000000, 128000000, 117000000, 109000000, 103000000, 104000000, 96000000, 115000000, 125000000, 141000000, 164000000, 240000000, 263000000, 286000000, 245000000, 192000000, 279000000, 289000000]
        },
        "MKE": {
            "City": "Milwaukee",
            "pop": [593253, 591706, 589519, 586414, 581735, 578531, 577723, 578730, 582866, 594883, 597132, 598627, 599997, 600775, 600607, 597179, 594154, 591961, 590952, 576301, 569330, 563305],
            "wagetax": [None] * 22,
            "pay": [56000000, 62000000, 57000000, 61000000, 67000000, 68000000, 66000000, 75000000, 69000000, 66000000, 54000000, 68000000, 69000000, 71000000, 80000000, 109000000, 127000000, 126000000, 125000000, 121000000, 151000000, 188000000],
            "salestax": [None] * 22,
            "NBArev": [67000000, 70000000, 77000000, 78000000, 87000000, 88000000, 94000000, 91000000, 92000000, 92000000, 87000000, 109000000, 110000000, 126000000, 146000000, 179000000, 204000000, 283000000, 239000000, 212000000, 352000000, 329000000]
        },
        "TOR": {
            "City": "Toronto",
            "pop": [2583940, 2602400, 2592910, 2587190, 2588700, 2608510, 2612640, 2626300, 2649010, 2675210, 2704880, 2737700, 2764900, 2789540, 2798440, 2819400, 2862660, 2917920, 2963230, 2984060, 2955860, 3025650],
            "wagetax": [None] * 22,
            "pay": [53000000, 58000000, 59000000, 65000000, 68000000, 57000000, 70000000, 78000000, 72000000, 75000000, 49000000, 73000000, 81000000, 87000000, 121000000, 127000000, 146000000, 115000000, 108000000, 108000000, 137000000, 156000000],
            "salestax": [None] * 22,
            "NBArev": [87000000, 96000000, 100000000, 94000000, 105000000, 124000000, 138000000, 133000000, 138000000, 134000000, 121000000, 149000000, 151000000, 163000000, 193000000, 250000000, 275000000, 334000000, 264000000, 194000000, 299000000, 305000000]
        },
        "SPO": {
            "City": "Spokane",
            "pop": [196159, 196801, 196826, 198037, 198300, 198979, 201204, 202323, 203365, 209537, 209079, 209371, 210131, 211218, 212695, 215298, 217439, 219268, 221804, 228850, 229071, 230160],
            "wagetax": [None] * 22,
            "pay": [None] * 22,
            "salestax": [29393967, 29055968, 29752334, 29919893, 31298182, 37099047, 39187817, 37999000, 35403343, 35297900, 35806600, 41702800, 44870900, 48085600, 50176400, 53232000, 57325000, 60329000, 63792000, 59684000, 76997000, 85720000],
            "NBArev": [None] * 22
        },
        "TAC": {
            "City": "Tacoma",
            "pop": [194978, 196713, 195635, 194577, 193936, 194880, 195176, 197653, 200084, 198324, 199635, 201687, 202295, 203929, 207010, 210610, 213776, 216495, 218098, 219383, 219205, 221776],
            "wagetax": [None] * 22,
            "pay": [None] * 22,
            "salestax": [31511000, 33915000, 43528000, 43842000, 49272000, 54071001, 54508000, 47681000, 42256000, 41942000, 42643000, 46738000, 45743000, 47976000, 81772000, 58970000, 71634000, 76153000, 81132000, 77718000, 106635000, 120772000],
            "NBArev": [None] * 22
        },
        "DEN": {
            "City": "Denver",
            "pop": [561976, 556790, 552588, 550756, 551691, 556895, 564395, 575721, 589011, 603012, 620250, 634965, 649270, 664540, 683328, 696273, 704869, 716340, 725508, 735538, 711463, 713252],
            "wagetax": [None] * 22,
            "pay": [None] * 22,
            "salestax": [355443000, 375334000, 366627000, 361988000, 417079000, 420693000, 455436000, 468137000, 421838000, 447071000, 481023000, 494495000, 539348000, 608307000, 640251000, 676916000, 721512000, 762201000, 896924000, 791509000, 1079107000, 1225079000], # 2001 value missing. interpolated
            "NBArev": [75000000, 75000000, 89000000, 94000000, 100000000, 104000000, 112000000, 115000000, 113000000, 113000000, 110000000, 124000000, 136000000, 140000000, 157000000, 202000000, 222000000, 252000000, 218000000, 185000000, 273000000, 348000000],

        },
        "BTR": {
            "City": "Baton Rouge",
            "pop": [224833, 222908, 222782, 222732, 220899, 228866, 225522, 223823, 223930, 229239, 228527, 228805, 228284, 227815, 227191, 226712, 224441, 222379, 220787, 224480, 222185, 221453],
            "wagetax": [None] * 22,
            "pay": [None] * 22,
            "salestax": [77319666, 77534525, 77644392, 78348495, 78508665, 85038718, 87494793, 88998396, 90799213, 93493918, 97069860, 97484337, 98807202, 99525928, 104103683, 108105130, 108487127, 109113017, 109938263, 111616795, 125102759, 136626781], # 2004 value missing. interpolated
            "NBArev": [None] * 22
    }
    },
    "National": {
        "CPI": [
            100, 101.59567, 103.93034, 106.70275, 110.29419, 113.84796, 117.11603, 121.58395, 121.19445, 123.17788, 127.04523, 129.67912, 131.58018, 133.70581, 133.86778, 135.56437, 138.45385, 141.83074, 144.40249, 146.20635, 153.05201, 165.28345
        ],
        "PCE": [
            100, 101.31338, 103.44169, 106.00946, 109.06383, 112.1349, 115.01231, 118.41583, 118.08571, 120.20075, 123.24161, 125.5398, 127.19513, 128.9754, 129.21168, 130.52007, 132.79867, 135.51683, 137.46421, 138.95344, 144.75197, 154.10264
        ],
        "GDP": [
            100.0, 103.3, 108.3, 115.5, 123.2, 130.6, 136.8, 139.6, 136.8, 142.2, 147.4, 153.6, 159.5, 166.4, 172.9, 177.7, 185.3, 195.2, 203.4, 201.5, 223.0, 243.3
        ]
    }
}

# Step 8: Define functions to determine NBArev presence, wage tax presence, sales tax presence, and add NBA revenue flag

def set_group_presence(data):
    NBArev_cities = ['PHL', 'CLE', 'DET', 'MIA', 'MEM', 'DAL', 'HOU', 'SAC', 'MKE', 'TOR', 'DEN']
    wagetax_cities = ['PHL', 'CLE', 'DET', 'STL', 'PIT', 'CIN', 'BAL']
    salestax_cities = ['PHL', 'DET', 'MIA', 'SAC', 'HOU', 'SPO', 'TAC', 'DEN', 'BTR']

    # Set flags for NBA revenue presence, wage tax, and sales tax
    for city_code, city_info in data['Cities'].items():
        city_info['has_NBArev'] = 1 if city_code in NBArev_cities else 0
        city_info['has_wagetax'] = 1 if city_code in wagetax_cities else 0
        city_info['has_salestax'] = 1 if city_code in salestax_cities else 0

set_group_presence(data)

# Step 9: Convert dictionary to DataFrame
data_list = [
    {
        "Year": pd.to_datetime(year, format='%Y'),
        "City": city_code,
        "Population": city_info.get("pop", [None] * len(data["YEARS"]))[min(data["YEARS"].index(year), len(city_info.get("pop", []))-1)],
        "wagetax": (city_info.get("wagetax") if isinstance(city_info.get("wagetax"), list) else [None] * len(data["YEARS"]))[min(data["YEARS"].index(year), len(city_info.get("wagetax", []))-1)],
        "salestax": (city_info.get("salestax") if isinstance(city_info.get("salestax"), list) else [None] * len(data["YEARS"]))[min(data["YEARS"].index(year), len(city_info.get("salestax", []))-1)],
        "NBArev": (city_info.get("NBArev") if isinstance(city_info.get("NBArev"), list) else [None] * len(data["YEARS"]))[min(data["YEARS"].index(year), len(city_info.get("NBArev", []))-1)],
        "pay": city_info.get("pay", [None] * len(data["YEARS"]))[min(data["YEARS"].index(year), len(city_info.get("pay", []))-1)],
        "CPI": data["National"]["CPI"][min(data["YEARS"].index(year), len(data["National"]["CPI"])-1)],
        "GDP": data["National"]["GDP"][min(data["YEARS"].index(year), len(data["National"]["GDP"])-1)],
        "PCE": data["National"]["PCE"][min(data["YEARS"].index(year), len(data["National"]["PCE"])-1)],
        "has_NBArev": city_info['has_NBArev'],
        "has_wagetax": city_info['has_wagetax'],
        "has_salestax": city_info['has_salestax'],
        "Intervention_2016": 1 if year == 2016 else 0,
        "post_2016": 1 if year >= 2016 else 0
    }
    for year in data["YEARS"] for city_code, city_info in data["Cities"].items()
]

df = pd.DataFrame(data_list)

# Step 10: Verify the correct assignment of flags
print(df[['City', 'has_NBArev', 'has_wagetax', 'has_salestax']].drop_duplicates())

# Step 11: Smoothing pandemic effects
def smooth_pandemic_effects(data):
    indices = {year: data['YEARS'].index(year) for year in [2020, 2021] if year in data['YEARS']}
    for city_code, city_info in data['Cities'].items():
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

smooth_pandemic_effects(data)

# Step 12: Adjust for inflation
def adjust_for_inflation(data):
    cpi_index = {year: cpi for year, cpi in zip(data['YEARS'], data['National']['CPI'])}
    for city_code, city_info in data['Cities'].items():
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

adjust_for_inflation(data)

# Step 13: adjust for population
def adjust_for_per_capita(data):
    for city_code, city_info in data['Cities'].items():
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

adjust_for_per_capita(data)

# Step 14: Adjust for population
def adjust_for_per_capita(data):
    for city_code, city_info in data['Cities'].items():
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

adjust_for_per_capita(data)

# Step 15: Graphically show intervention with aggregated lines for NBA revenue, sales tax, and wage tax
# Extract years
years = data['YEARS']

# Cities to include in the aggregation
cities_to_aggregate = ['PHL', 'CLE', 'DET', 'MIA', 'MEM', 'DAL', 'HOU', 'SAC', 'MKE', 'TOR', 'DEN']

# Initialize lists to store aggregated values
nba_rev_agg = []
sales_tax_agg = []
wage_tax_agg = []

# Calculate aggregated values for each year
for i in range(len(years)):
    nba_rev_total = sum(data['Cities'][city]['NBArev'][i] for city in cities_to_aggregate if data['Cities'][city]['NBArev'][i] is not None)
    sales_tax_total = sum(data['Cities'][city]['salestax'][i] for city in cities_to_aggregate if data['Cities'][city]['salestax'][i] is not None)
    wage_tax_total = sum(data['Cities'][city]['wagetax'][i] for city in cities_to_aggregate if data['Cities'][city]['wagetax'][i] is not None)
    
    nba_rev_agg.append(nba_rev_total)
    sales_tax_agg.append(sales_tax_total)
    wage_tax_agg.append(wage_tax_total)

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
        data['Cities'][city]['salestax'][years.index(year)]
        for city in cities_with_nba_sales_tax if data['Cities'][city]['salestax'][years.index(year)] is not None
    )
    sales_tax_non_nba_total = sum(
        data['Cities'][city]['salestax'][years.index(year)]
        for city in cities_without_nba_sales_tax if data['Cities'][city]['salestax'][years.index(year)] is not None
    )
    sales_tax_nba.append(sales_tax_nba_total)
    sales_tax_non_nba.append(sales_tax_non_nba_total)


# Fit linear regression models to the sales tax data
X_years_analysis = np.array(years_analysis).reshape(-1, 1)
model_nba_sales = LinearRegression().fit(X_years_analysis, sales_tax_nba)
model_non_nba_sales = LinearRegression().fit(X_years_analysis, sales_tax_non_nba)

# Predict sales tax using the fitted models
sales_tax_nba_pred = model_nba_sales.predict(X_years_analysis)
sales_tax_non_nba_pred = model_non_nba_sales.predict(X_years_analysis)

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
        data['Cities'][city]['wagetax'][years.index(year)]
        for city in cities_with_nba_wage_tax if data['Cities'][city]['wagetax'][years.index(year)] is not None
    )
    wage_tax_without_nba_total = sum(
        data['Cities'][city]['wagetax'][years.index(year)]
        for city in cities_with_wage_no_nba if data['Cities'][city]['wagetax'][years.index(year)] is not None
    )
    wage_tax_with_nba.append(wage_tax_with_nba_total)
    wage_tax_without_nba.append(wage_tax_without_nba_total)

# Fit linear regression models to the wage tax data
model_with_nba_wage = LinearRegression().fit(X_years_analysis, wage_tax_with_nba)
model_without_nba_wage = LinearRegression().fit(X_years_analysis, wage_tax_without_nba)

# Predict wage tax using the fitted models
wage_tax_with_nba_pred = model_with_nba_wage.predict(X_years_analysis)
wage_tax_without_nba_pred = model_without_nba_wage.predict(X_years_analysis)

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

#17 multivariate DID w Weighted Least Squares (WLS)

# Prepare the data for the combined sales and wage tax DiD analysis (2016 intervention year)
years_analysis = [year for year in years if year <= 2016]
data_list_combined = []

for year in years_analysis:
    for city_code, city_info in data['Cities'].items():
        nba_team = 1 if city_info['has_NBArev'] else 0
        salestax = city_info['salestax'][years.index(year)] if 'salestax' in city_info else None
        wagetax = city_info['wagetax'][years.index(year)] if 'wagetax' in city_info else None
        post_2016 = 1 if year >= 2016 else 0
        
        if salestax is not None or wagetax is not None:
            data_list_combined.append({
                'Year': year,
                'City': city_code,
                'SalesTax': salestax,
                'WageTax': wagetax,
                'NBA_Team': nba_team,
                'Post_2016': post_2016,
                'Interaction_2016': nba_team * post_2016
            })

combined_tax_did_df = pd.DataFrame(data_list_combined)

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

# Step 18: Diagnostics. Multicollinearity, Normality, Autorrelation and Heteroscedasticity Check for Both Models

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

# Autocorrelation Test (Durbin-Watson) for sales tax model
dw_test_sales_tax = durbin_watson(residuals_sales_tax)
print(f"Durbin-Watson test for autocorrelation (Sales Tax): statistic = {dw_test_sales_tax}")
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
print("\n")

# Autocorrelation Test (Durbin-Watson) for wage tax model
dw_test_wage_tax = durbin_watson(residuals_wage_tax)
print(f"Durbin-Watson test for autocorrelation (Wage Tax): statistic = {dw_test_wage_tax}")
print("\n")

## Step 19: Stationary Bootstraping

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

# Step 20: Post-Bootstrapping Diagnostics

# Re-running diagnostics to validate improvements and robustness

# Multicollinearity Test (VIF) for sales tax model
print("Post-Bootstrapping Variance Inflation Factor (VIF) for Sales Tax Model:")
vif_data_sales = pd.DataFrame()
vif_data_sales["Feature"] = X_sales.columns
vif_data_sales["VIF"] = [variance_inflation_factor(X_sales.values, i) for i in range(X_sales.shape[1])]
print(vif_data_sales)
print("\n")

# Normality Test (Shapiro-Wilk) for sales tax model
shapiro_test_sales_tax = shapiro(residuals_sales_tax)
print(f"Post-Bootstrapping Shapiro-Wilk test for normality (Sales Tax): p-value = {shapiro_test_sales_tax[1]}")
print("\n")

# Heteroscedasticity Test (Breusch-Pagan) for sales tax model
bp_test_sales_tax = het_breuschpagan(residuals_sales_tax, X_sales)
print(f"Post-Bootstrapping Breusch-Pagan test for heteroscedasticity (Sales Tax): p-value = {bp_test_sales_tax[1]}")
print("\n")

# Autocorrelation Test (Durbin-Watson) for sales tax model
dw_test_sales_tax = durbin_watson(residuals_sales_tax)
print(f"Post-Bootstrapping Durbin-Watson test for autocorrelation (Sales Tax): statistic = {dw_test_sales_tax}")
print("\n")

# Multicollinearity Test (VIF) for wage tax model
print("Post-Bootstrapping Variance Inflation Factor (VIF) for Wage Tax Model:")
vif_data_wage = pd.DataFrame()
vif_data_wage["Feature"] = X_wage.columns
vif_data_wage["VIF"] = [variance_inflation_factor(X_wage.values, i) for i in range(X_wage.shape[1])]
print(vif_data_wage)
print("\n")

# Normality Test (Shapiro-Wilk) for wage tax model
shapiro_test_wage_tax = shapiro(residuals_wage_tax)
print(f"Post-Bootstrapping Shapiro-Wilk test for normality (Wage Tax): p-value = {shapiro_test_wage_tax[1]}")
print("\n")

# Heteroscedasticity Test (Breusch-Pagan) for wage tax model
bp_test_wage_tax = het_breuschpagan(residuals_wage_tax, X_wage)
print(f"Post-Bootstrapping Breusch-Pagan test for heteroscedasticity (Wage Tax): p-value = {bp_test_wage_tax[1]}")
print("\n")

# Autocorrelation Test (Durbin-Watson) for wage tax model
dw_test_wage_tax = durbin_watson(residuals_wage_tax)
print(f"Post-Bootstrapping Durbin-Watson test for autocorrelation (Wage Tax): statistic = {dw_test_wage_tax}")
print("\n")
