import pandas as pd
from typing import Tuple

class DataLoader:
    def __init__(self, customers_file: str, noncustomers_file: str, actions_file: str):
        """
        Initialize the DataLoader with file paths for the datasets.
        """
        self.customers_file = customers_file
        self.noncustomers_file = noncustomers_file
        self.actions_file = actions_file

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load data from the provided file paths.
        
        Returns:
            Tuple of pandas DataFrames: (customers, noncustomers, actions)
        """
        customers = pd.read_csv(self.customers_file)
        noncustomers = pd.read_csv(self.noncustomers_file)
        actions = pd.read_csv(self.actions_file)
        return customers, noncustomers, actions

    def preprocess_customers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the customers dataset by handling missing values and converting data types.
        """
        df['CLOSEDATE'] = pd.to_datetime(df['CLOSEDATE'], errors='coerce')  # Convert CLOSEDATE to datetime
        df['MRR'] = pd.to_numeric(df['MRR'], errors='coerce')  # Ensure MRR is numeric
        df['EMPLOYEE_RANGE'] = df['EMPLOYEE_RANGE'].fillna("Unknown")  # Fill missing employee ranges
        df['INDUSTRY'] = df['INDUSTRY'].fillna("Unknown")  # Fill missing industries
        df['ALEXA_RANK'] = pd.to_numeric(df['ALEXA_RANK'], errors='coerce')  # Ensure MRR is numeric

        return df

    def preprocess_noncustomers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the noncustomers dataset by handling missing values and converting data types.
        """
        df['EMPLOYEE_RANGE'] = df['EMPLOYEE_RANGE'].fillna("Unknown")  # Fill missing employee ranges
        df['INDUSTRY'] = df['INDUSTRY'].fillna("Unknown")  # Fill missing industries
        df['ALEXA_RANK'] = pd.to_numeric(df['ALEXA_RANK'], errors='coerce')  # Ensure MRR is numeric

        return df

    def preprocess_actions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the actions dataset by handling missing values and converting data types.
        """
        df['WHEN_TIMESTAMP'] = pd.to_datetime(df['WHEN_TIMESTAMP'], errors='coerce')  # Convert timestamp to datetime
        df.fillna(0, inplace=True)  # Fill all missing values with 0
        return df

    def merge_datasets(self, customers: pd.DataFrame, noncustomers: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
        """
        Merge customers and noncustomers with the actions dataset on the 'id' column.
        
        Returns:
            Merged DataFrame.
        """
        # Add a flag to distinguish customers from noncustomers
        customers['IS_CUSTOMER'] = 1
        noncustomers['IS_CUSTOMER'] = 0
        noncustomers['MRR'] = 0
        # Combine customers and noncustomers into one dataset
        customer_f = customers.merge(actions, left_on='id', right_on='id')
        customer_f = customer_f[customer_f.CLOSEDATE>customer_f.WHEN_TIMESTAMP]
        noncustomer_f = noncustomers.merge(actions, left_on='id', right_on='id')
        customers_full = pd.concat([noncustomers, customers.drop(columns=['CLOSEDATE'])])
        c = pd.concat([customers_full['id'], pd.get_dummies(customers_full['INDUSTRY'], prefix = 'INDUSTRY'),\
            pd.get_dummies(customers_full['EMPLOYEE_RANGE'], prefix = 'EM')], axis=1)
        
        all_customers = pd.concat([customer_f.drop(columns=['CLOSEDATE']), noncustomer_f], ignore_index=True)

        # Create dummy variables for 'INDUSTRY'
        
        #industry_dummies = pd.get_dummies(all_customers['INDUSTRY'], prefix='INDUSTRY')
        # industry_dummies_non = pd.get_dummies(noncustomer_f['INDUSTRY'], prefix='INDUSTRY')


        # # Add dummy variables to the DataFrame
        # customer_f = pd.concat([customer_f, industry_dummies], axis=1)
        # noncustomer_f = pd.concat([noncustomer_f, industry_dummies_non], axis=1)

        # Select numeric columns including the new dummy variables; # Merge with actions
        numeric_columns = all_customers.drop(columns=['id', 'MRR']).select_dtypes(include='number').columns

        # Group by 'id' and take the mean
        #
        companies = all_customers.groupby('id')[numeric_columns].mean().reset_index()
        companies_f = companies.merge(c, left_on= 'id', right_on ='id', how='left')
        # companies_f = pd.concat([customer_f, industry_dummies], axis=1)

        return companies_f

    def load_and_preprocess(self) -> pd.DataFrame:
        """
        Load and preprocess all datasets, then merge them into a single DataFrame.
        
        Returns:
            Final merged and cleaned DataFrame.
        """
        customers, noncustomers, actions = self.load_data()
        customers = self.preprocess_customers(customers)
        noncustomers = self.preprocess_noncustomers(noncustomers)
        actions = self.preprocess_actions(actions)
        merged_data = self.merge_datasets(customers, noncustomers, actions)
        return merged_data
