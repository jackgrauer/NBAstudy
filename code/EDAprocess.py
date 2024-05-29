import matplotlib as plt
import pandas as pd
from code.EDAraw import EDA

def process_EDA():
    # Initialize DataFrame with provided data
    df = pd.DataFrame(EDA)
    
    # Ensure all necessary columns exist in the DataFrame
    required_columns = [
        'BIRT_Gross_Receipts_Rate', 'BIRT_Net_Income_Rate', 'Wage_Tax_Rate_Resident',
        'Wage_Tax_Rate_Non_Resident', 'Sales_Tax_Rate', 'Net_Profit_Tax_Rate_Resident',
        '76ers_Gate_Receipts', 'Amusement_Tax_Rate', '76ers_Revenue', 
        '76ers_Operating_Income', '76ers_Player_Expenses', '76ers_PILOT', 
        'PHL_Business_Priv', 'PHL_Wage_and_Earnings', 'PHL_Sales_and_Use', 'Year'
    ]

    for col in required_columns:
        if col not in df.columns:
            df[col] = 0  # Assign a default value if the column is missing

    # Use original tax rates without inflation adjustments
    df['76ers_BIRT_Gross_Receipts_Rate'] = df['BIRT_Gross_Receipts_Rate']
    df['76ers_BIRT_Net_Income_Rate'] = df['BIRT_Net_Income_Rate']
    df['76ers_Wage_Tax_Rate_Resident'] = df['Wage_Tax_Rate_Resident']
    df['76ers_Wage_Tax_Rate_Non_Resident'] = df['Wage_Tax_Rate_Non_Resident']
    df['76ers_Sales_Tax_Rate'] = df['Sales_Tax_Rate']
    df['76ers_Net_Profit_Tax_Rate_Resident'] = df['Net_Profit_Tax_Rate_Resident']

    # Calculate tax revenue contributions using the original tax rates
    df['76ers_Amusement_Tax_Contribution'] = df['76ers_Gate_Receipts'] * df['Amusement_Tax_Rate']
    df['76ers_BIRT_Gross_Receipts_Contribution'] = df['76ers_Revenue'] * df['76ers_BIRT_Gross_Receipts_Rate']
    df['76ers_BIRT_Net_Income_Contribution'] = df['76ers_Operating_Income'] * df['76ers_BIRT_Net_Income_Rate']
    df['76ers_Wage_Tax_Contribution_Resident'] = df['76ers_Player_Expenses'] * (1/3) * df['76ers_Wage_Tax_Rate_Resident']
    df['76ers_Wage_Tax_Contribution_NonResident'] = df['76ers_Player_Expenses'] * (2/3) * df['76ers_Wage_Tax_Rate_Non_Resident']
    df['76ers_Sales_Tax_Contribution'] = df['76ers_Revenue'] * df['76ers_Sales_Tax_Rate']
    df['76ers_Net_Profits_Tax_Contribution'] = df['76ers_Operating_Income'] * df['76ers_Net_Profit_Tax_Rate_Resident']

    # Combine BIRT Contributions and Wage Tax Contributions for 76ers
    df['76ers_Total_BIRT_Contribution'] = df['76ers_BIRT_Gross_Receipts_Contribution'] + df['76ers_BIRT_Net_Income_Contribution']
    df['76ers_Total_Wage_Tax_Contribution'] = df['76ers_Wage_Tax_Contribution_Resident'] + df['76ers_Wage_Tax_Contribution_NonResident']

    # Calculate total tax revenue per year from 76ers
    df['76ers_Total_Revenue_Per_Year'] = df[['76ers_Amusement_Tax_Contribution', '76ers_BIRT_Gross_Receipts_Contribution', '76ers_BIRT_Net_Income_Contribution', '76ers_Wage_Tax_Contribution_Resident', '76ers_Wage_Tax_Contribution_NonResident', '76ers_Sales_Tax_Contribution', '76ers_Net_Profits_Tax_Contribution', '76ers_PILOT']].sum(axis=1)

    # Clip negative values to 0
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

    # Print yearly tax contributions from the 76ers
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

    # Print Philadelphia tax revenues by year
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

    # Visualization of tax contributions and revenues
    fig, ax = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Plot Philadelphia tax revenues
    ax[0].plot(df['Year'], df['PHL_Business_Priv'], label='Business Privilege', marker='o')
    ax[0].plot(df['Year'], df['PHL_Wage_and_Earnings'], label='Wage and Earnings', marker='o')
    ax[0].plot(df['Year'], df['PHL_Sales_and_Use'], label='Sales and Use', marker='o')
    ax[0].axvline(x=2016, color='red', linestyle='--', label='2016 Intervention')
    ax[0].set_title('Philadelphia Tax Revenues by Year')
    ax[0].set_ylabel('Revenue ($)')
    ax[0].legend()

    # Plot 76ers tax contributions
    ax[1].plot(df['Year'], df['76ers_Total_Revenue_Per_Year'], label='Total 76ers Tax Contributions', color='green', marker='o')
    ax[1].axvline(x=2016, color='red', linestyle='--', label='2016 Intervention')
    ax[1].set_title('76ers Tax Contributions by Year')
    ax[1].set_xlabel('Year')
    ax[1].set_ylabel('Contributions ($)')
    ax[1].legend()

    plt.tight_layout()
    plt.show()

    return df