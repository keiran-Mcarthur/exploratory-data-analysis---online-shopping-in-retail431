import pandas as pd
def csv_to_pd_df():
    customer_activity_df = pd.read_csv('customer_activity.csv')
    shape =  customer_activity_df.shape
    print(f'This dataset has {shape[0]} rows and {shape[1]} columns')
    print(customer_activity_df)

csv_to_pd_df()