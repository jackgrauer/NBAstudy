import pandas as pd
import matplotlib.pyplot as plt
from code.EDAraw import EDA

def process_EDA():
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

    # Define headers with each word on a new line to fit vertically
    headers = [
        "Year", "Amusement\nTax", "BIRT\nGross\nReceipts", "BIRT\nNet\nIncome",
        "Wage\nTax\nResident", "Wage\nTax\nNon-Res.", "Sales\nTax",
        "Net\nProfits\nTax", "PILOT", "Total\nRevenue"
    ]

    # Adjust column widths to accommodate vertical headers
    header_format = "| {:<4} | {:<9} | {:<12} | {:<12} | {:<11} | {:<14} | {:<9} | {:<14} | {:<5} | {:<13} |"
    row_format = "| {:<4} | {:<9} | {:<12} | {:<12} | {:<11} | {:<14} | {:<9} | {:<14} | {:<5} | {:<13} |"

    # Print headers
    for i in range(max(map(lambda x: x.count("\n"), headers)) + 1):
        print("| {:<4} | {:<9} | {:<12} | {:<12} | {:<11} | {:<14} | {:<9} | {:<14} | {:<5} | {:<13} |".format(
            *[header.split("\n")[i] if len(header.split("\n")) > i else "" for header in headers]
        ))

    # Print divider
    print("-" * 129)

    # Print data rows
    for index, row in df.iterrows():
        print(row_format.format(
            int(row['Year']),
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
        print("-" * 129)








    # Print Philadelphia tax revenues by year
    print("Philadelphia Tax Revenues By Year:\n")
    headers = ["Year", "Business Privilege", "Wage and Earnings", "Sales and Use"]
    header_format = "| {:<6} | {:<20} | {:<20} | {:<20} |"
    row_format = "| {:<6} | {:<20} | {:<20} | {:<20} |"

    print(header_format.format(*headers))
    print("-" * 75)

    for index, row in df.iterrows():
        print(row_format.format(
            int(row['Year']),
            int(row['PHL_Business_Priv']),
            int(row['PHL_Wage_and_Earnings']),
            int(row['PHL_Sales_and_Use'])
        ))
        print("-" * 75)



    # Visualization of tax contributions and revenues
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Plot Philadelphia tax revenues
    axes[0].plot(df['Year'], df['PHL_Business_Priv'], label='Business Privilege', marker='o')
    axes[0].plot(df['Year'], df['PHL_Wage_and_Earnings'], label='Wage and Earnings', marker='o')
    axes[0].plot(df['Year'], df['PHL_Sales_and_Use'], label='Sales and Use', marker='o')
    axes[0].axvline(x=2016, color='red', linestyle='--', label='2016 Intervention')
    axes[0].set_title('Philadelphia Tax Revenues by Year')
    axes[0].set_ylabel('Revenue ($)')
    axes[0].legend()

    # Plot 76ers tax contributions
    axes[1].plot(df['Year'], df['76ers_Total_Revenue_Per_Year'], label='Total 76ers Tax Contributions', color='green', marker='o')
    axes[1].axvline(x=2016, color='red', linestyle='--', label='2016 Intervention')
    axes[1].set_title('76ers Tax Contributions by Year')
    axes[1].set_xlabel('Year')
    axes[1].set_ylabel('Contributions ($)')
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    return df
