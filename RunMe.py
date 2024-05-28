# Import necessary modules
from code.EDAprocess import process_EDA

def main():
    # Run the EDA process
    EDA_df = process_EDA()

    # Optionally, print the dataframe to verify
    print(EDA_df)

if __name__ == "__main__":
        main()


        