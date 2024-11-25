from data.loader import DataLoader

# File paths
customers_file = "./data/customers.csv"
noncustomers_file = "./data/noncustomers.csv"
actions_file = "./data/usage_actions.csv"

# Initialize loader
loader = DataLoader(customers_file, noncustomers_file, actions_file)

# Load and preprocess data
merged_data = loader.load_and_preprocess()

# Display the processed data
print(merged_data.head())