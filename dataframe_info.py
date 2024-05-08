import pandas as pd
import numpy as np
from matplotlib import pyplot
class DataFrameInfo():
    def __init__(self,customer_activity_df):
         self. customer_activity_df = customer_activity_df
    
    def describe(self):
        print(round(customer_activity_df.describe(),1))
        customer_activity_df.info()
        shape = customer_activity_df.shape
        print(f'This dataset has {shape[0]} rows and {shape[1]} columns')
    
    def statistics_extraction(self):
        dataframe_means = customer_activity_df[['administrative', 'administrative_duration', 'informational', 'informational_duration', 'product_related', 'product_related_duration', 'bounce_rates', 'exit_rates', 'page_values']].mean()
        print("The mean of the numeric columns are")
        print(round(dataframe_means,1))
        dataframe_median = customer_activity_df[['administrative', 'administrative_duration', 'informational', 'informational_duration', 'product_related', 'product_related_duration', 'bounce_rates', 'exit_rates', 'page_values']].median()
        print("The median of the numeric columns are" )
        print(round(dataframe_median,1))
        selected_columns = ['administrative', 'administrative_duration', 'informational', 'informational_duration', 'product_related', 'product_related_duration', 'bounce_rates', 'exit_rates', 'page_values']
        dataframe_std = customer_activity_df[selected_columns].std()
        print("The standard deviations of the numeric columns are:")
        print(round(dataframe_std, 1))
    
    def category_distinct_values(self):
        for c in customer_activity_df.columns:
            if customer_activity_df[c].dtype == 'object' or 'boolean':
              print ("---- %s ---" % c)
              print (customer_activity_df[c].value_counts())
    
    def null_values(self):
        null_count = customer_activity_df.isna().sum()  
        print(null_count) 
        null_as_a_precentage = customer_activity_df.isna().mean()*100
        print(round(null_as_a_precentage,1))
    
    
    
    
customer_activity_df = pd.read_csv('customer_activity.csv')
test = DataFrameInfo(customer_activity_df) 
test.statistics_extraction()
test.category_distinct_values()
test.null_values()


