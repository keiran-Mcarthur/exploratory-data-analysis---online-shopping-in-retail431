import pandas as pd 
from scipy.stats import normaltest
from statsmodels.graphics.gofplots import qqplot
from matplotlib import pyplot
import numpy as np
import seaborn as sns
import plotly.express as px
class plotter():
    def __init__(self,customer_activity_df):
         self. customer_activity_df = customer_activity_df
    
    
    def visualise_null_values(self):
      null_count = self.customer_activity_df.isnull().sum()
      pyplot.figure(figsize=(10, 6))
      pyplot.barh(null_count.index, null_count, color='orange', label='After Removal')
      pyplot.xlabel('Number of NULL values')
      pyplot.ylabel('Columns')
      pyplot.title('Removal of NULL Values')
      pyplot.legend()
      pyplot.show()

    def skewed_data(self):
     df = customer_activity_df
     df['exit_rates'].hist(bins=40)
     pyplot.title('Skewed data')
     pyplot.show()
     print(f"Skew of administrative column is {df['informational'].skew()}")
     qq_plot = qqplot(customer_activity_df['product_related_duration'] , scale=1 ,line='q')
     pyplot.title('Skewed data')
     pyplot.show()
    
    def outliers_detection_visual(self):
     numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
     for c in [c for c in customer_activity_df.columns if customer_activity_df[c].dtype in numerics]:
      customer_activity_df.hist(column=[c], bins=40)
      pyplot.title('Histogram of {}'.format(c))
      pyplot.show()
     numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
     for c in [c for c in customer_activity_df.columns if customer_activity_df[c].dtype in numerics]:
      pyplot.figure()
      customer_activity_df.boxplot(column=[c], showfliers=True, whis=1.5)
      pyplot.title('Box plot with whiskers and scatter points of {}'.format(c))
      pyplot.show()
    
    def outliers_detection_statisical(self):
     Q1 = customer_activity_df['page_values'].quantile(0.25)
     Q3 = customer_activity_df['page_values'].quantile(0.75)
     IQR = Q3 - Q1
     lower_bound = Q1 - 1.5 * IQR
     print(f"Q1 (25th percentile): {Q1}")
     upper_bound = Q3 + 1.5 * IQR
     print(f"Q3 (75th percentile): {Q3}")
     print(f"IQR: {IQR}")
     outliers =  (customer_activity_df['page_values'] < lower_bound) | (customer_activity_df['page_values'] > upper_bound)
     print("Outliers:", customer_activity_df['page_values'][outliers])
     print(customer_activity_df['page_values'])
     print(customer_activity_df['page_values'].value_counts())
    
    def frequnecy_tables(self):
      category = ['category', 'object']
      for c in [c for c in customer_activity_df.columns if customer_activity_df[c].dtype in category]:
          frequency_table = customer_activity_df[c].value_counts()
          frequency_table.plot(kind='bar', column=[c])
          pyplot.xlabel('Categories')
          pyplot.ylabel('Frequency')
          pyplot.title('Frequncy table of {}'.format(c))
          pyplot.show()
    
   
    
class DataFrameTransform():
     def __init__(self,customer_activity_df):
         self. customer_activity_df = customer_activity_df
     
     def null_values(self):
      null_count = customer_activity_df.isna().sum()  
      print(null_count) 
      null_as_a_precentage = customer_activity_df.isna().mean()*100
      print(round(null_as_a_precentage,1))
    
     def normality_test(self):
        data = customer_activity_df['page_values']
        stat, p = normaltest(data, nan_policy='omit')
        print('Statistics=%.3f, p=%.3f' % (stat, p))
        customer_activity_df['page_values'].hist(bins=7,edgecolor='black')
        pyplot.show() 
        qq_plot = qqplot(customer_activity_df['page_values'] , scale=1 ,line='q')
        pyplot.show()

     def impute_nulls(self):
          customer_activity_df['administrative'] = customer_activity_df['administrative'].fillna(customer_activity_df['administrative'].mean())
          customer_activity_df['administrative_duration'] = customer_activity_df['administrative_duration'].fillna(customer_activity_df['administrative_duration'].median())
          customer_activity_df['informational_duration'] = customer_activity_df['informational_duration'].fillna(customer_activity_df['informational_duration'].median())
          customer_activity_df['product_related'] = customer_activity_df['product_related'].fillna(customer_activity_df['product_related'].median())
          customer_activity_df['product_related_duration'] = customer_activity_df['product_related_duration'].fillna(customer_activity_df['product_related_duration'].median())
          mode = customer_activity_df['operating_systems'].mode()[0]
          customer_activity_df['operating_systems'] = customer_activity_df['operating_systems'].fillna(mode)
     
     def skew_correction(self):
          small_constant = 0.001
          numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
          for c in [c for c in customer_activity_df.columns if customer_activity_df[c].dtype in numerics]:
            customer_activity_df[c] = np.log(customer_activity_df[c]+ small_constant)
            skewness_after_log_transform = customer_activity_df[c].skew()
            print("Skewness after log transform:", skewness_after_log_transform)
            sns.histplot(customer_activity_df[c], kde=True)
            pyplot.title('Dataset  After Log Transform')
            pyplot.show()
     
     
     def outliers_trimming(self):
          df_clean = customer_activity_df
          columns_to_clean = ['informational', 'informational_duration', 'page_values']
          columns_to_clean = ['informational', 'informational_duration', 'page_values']
          for column in columns_to_clean:
               Q1 = df_clean[column].quantile(0.25)
               Q3 = df_clean[column].quantile(0.75)
               IQR = Q3 - Q1
               lower_bound = Q1 - 1.5 * IQR
               upper_bound = Q3 + 1.5 * IQR
               lower_winsor = df_clean[column].quantile(0.05)
               upper_winsor = df_clean[column].quantile(0.95)
               df_clean[column] = np.where(df_clean[column] == lower_bound, lower_winsor, df_clean[column])
               df_clean[column] = np.where(df_clean[column] == upper_bound, upper_winsor, df_clean[column])
          return df_clean
    
     def outliers_removal(self):
      customer_activity_df_clean = customer_activity_df
      columns_to_clean = ['product_related', 'exit_rates']
      for column in columns_to_clean:
        Q1 = customer_activity_df_clean[column].quantile(0.25)
        Q3 = customer_activity_df_clean[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        Outlier = []
        for i in customer_activity_df_clean[column]:
            if i <= lower_bound or i >= upper_bound:
                Outlier.append(i)
                customer_activity_df_clean.drop(customer_activity_df_clean[customer_activity_df_clean[column] == i].index, inplace=True)
     
        Index_Outlier = customer_activity_df_clean[(customer_activity_df_clean[column] <= lower_bound) | (customer_activity_df_clean[column] >= upper_bound)].index
        customer_activity_df_clean.drop(Index_Outlier, inplace=True)
    
      return customer_activity_df_clean

     def product_related_duration_outliers_removal(self):
      customer_activity_df['product_related_duration'] = customer_activity_df['product_related_duration'][customer_activity_df['product_related_duration'] > -5]
      print(f"Skew of product_related_duration column is {customer_activity_df['product_related_duration'].skew()}")
    
     
     

     
     def collinearity(self):
      numeric_columns = customer_activity_df.select_dtypes(include=['number']).columns
      customer_activity_df_numeric = customer_activity_df[numeric_columns]
      correlation_matrix = customer_activity_df_numeric.corr()
      pyplot.figure(figsize=(12, 10))
      sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
      pyplot.title('Correlation Heatmap')
      pyplot.show()

      

    
   
customer_activity_df = pd.read_csv('customer_activity.csv')
test = plotter(customer_activity_df) 
test_impute = DataFrameTransform(customer_activity_df)
test_impute.null_values()
test_impute.normality_test()
test_impute.impute_nulls()
test.visualise_null_values()
test.skewed_data()
test_impute.skew_correction()
test.outliers_detection_visual()
test.outliers_detection_statisical()
test.frequnecy_tables()
test_impute.outliers_trimming()
test_impute.outliers_removal()
test_impute.product_related_duration_outliers_removal()
test.outliers_detection_visual()
test.outliers_detection_statisical()
test_impute.collinearity()