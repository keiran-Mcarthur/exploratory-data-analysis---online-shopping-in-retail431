from matplotlib import pyplot
from scipy.stats import normaltest
from statsmodels.graphics.gofplots import qqplot
import numpy as np
import pandas as pd 
import plotly.express as px
import seaborn as sns

class Plotter():
    """This class is called plotter.
    key functions:
    self. customer_activity_df = customer_activity_df -- used to create instance(self) which is used with customer_activity_df so can be used throughtout the class.
    """
    def __init__(self,customer_activity_df):
        self. customer_activity_df = customer_activity_df
    
    
    def visualise_null_values(self):
        """visualise null values in the dataset.
        Key functions:
        .isnull().sum() -- counts the null values and returns them as a sum.
        pyplot.barh() --plot the count of nulls as a barchart.
        """
        null_count = customer_activity_df.isnull().sum()
        pyplot.figure(figsize=(10, 6))
        pyplot.barh(null_count.index, null_count, color='orange', label='After Removal')
        pyplot.xlabel('Number of NULL values')
        pyplot.ylabel('Columns')
        pyplot.title('Removal of NULL Values')
        pyplot.legend()
        pyplot.show()

    def skewed_data(self):
        """Visualise the skew data in numeric columns.
        key functions:
        for loop -- Iterates through the columns that are in the numeric variables.
        customer_activity_df.hist() -- creates an histogram for each of the columns in the for loop.
        """
        numeric = [ 'int64', 'float64']
        for columns in [columns for columns in customer_activity_df.columns if customer_activity_df[columns].dtype in numeric]:
          customer_activity_df.hist(column=[columns], bins=40)
          pyplot.title('Histogram of Skew Data for {}'.format(columns))
          pyplot.show()
          print(f"Skew of administrative column is {customer_activity_df[columns].skew()}")
        qq_plot = qqplot(customer_activity_df['product_related_duration'] , scale=1 ,line='q')
        pyplot.title('QQ plot of Skew Data for Product Related Duration')
        pyplot.show()
    
    def outliers_detection_visual(self):
        """Visualises the outliers in numeric columns.
        key functions:
        for loop -- Iterates through the columns that are in the numeric variables.
        customer_activity_df.hist() -- creates an histogram for each of the columns in the for loop.
        customer_activity_df.boxplot() -- creates an boxplot for each of the columns in the for loop.
        """
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        for columns in [columns for columns in customer_activity_df.columns if customer_activity_df[columns].dtype in numerics]:
          customer_activity_df.hist(column=[columns], bins=40)
          pyplot.title('Histogram of outliers for  {}'.format(columns))
          pyplot.show()
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        for columns in [columns for columns in customer_activity_df.columns if customer_activity_df[columns].dtype in numerics]:
          pyplot.figure()
          customer_activity_df.boxplot(column=[columns], showfliers=True, whis=1.5)
          pyplot.title('Box plot with whiskers and scatter points of {}'.format(columns))
          pyplot.show()
    
    def outliers_detection_statisical(self):
        """Calculates IQR for each of the numeric columns.
        key functions:
        .quantile() -- used to calculate the first and third quantile for each column which can be used to detect outliers based on IQR. 
        """
        Q1 = customer_activity_df['page_values'].quantile(0.25)
        Q3 = customer_activity_df['page_values'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        print(f'Q1 (25th percentile): {Q1}')
        upper_bound = Q3 + 1.5 * IQR
        print(f'Q3 (75th percentile): {Q3}')
        print(f'IQR: {IQR}')
        outliers =  (customer_activity_df['page_values'] < lower_bound) | (customer_activity_df['page_values'] > upper_bound)
        print("Outliers:", customer_activity_df['page_values'][outliers])
        print(customer_activity_df['page_values'])
        print(customer_activity_df['page_values'].value_counts())
    
    def frequnecy_tables(self):
        """Visualise categorical columns.
        key functions:
        for loop -- Iterates through the columns that are in the category variable.
        customer_activity_df.hist() -- creates an histogram for each of the columns in the for loop.
        """
        category = ['category', 'object']
        for columns in [columns for columns in customer_activity_df.columns if customer_activity_df[columns].dtype in category]:
            frequency_table = customer_activity_df[columns].value_counts()
            frequency_table.plot(kind='bar', column=[columns])
            pyplot.xlabel('Categories')
            pyplot.ylabel('Frequency')
            pyplot.title('Frequncy table of {}'.format(columns))
            pyplot.show()
    
   
class DataFrameTransform():
     """This class is called DataFrameTransform.
     key functions:
     self. customer_activity_df = customer_activity_df -- used to create instance(self) which is used with customer_activity_df so can be used throughtout the class.
     self.df_copy_deep = self.create_copy_deep() -- which used to create an instance of the copied dataframe.
     """
     def __init__(self,customer_activity_df):
        self. customer_activity_df = customer_activity_df
        self.df_copy_deep = self.create_copy_deep()
     
     
     def null_values(self):
        """Calculates null values in dataset.
        Key functions:
        .isnull().sum() -- counts the null values and return them as a sum. 
        .isna().mean() * 100 -- converts the count of null values to a precentage.
        """
        null_count = customer_activity_df.isna().sum()  
        print(null_count) 
        null_as_a_precentage = customer_activity_df.isna().mean()*100
        print(round(null_as_a_precentage,1))
    
     def normality_test(self):
        """Test of normality in numeric columns.
        key functions:
        for loop -- iterates through the columns that are in the numeric variable.
        normaltest() -- calculates the p value for each of the variables that are a numeric data type.
        customer_activity_df.hist() -- creates an histogram for each of the columns in the for loop.
        """
        numeric = [ 'int64', 'float64']
        for columns in [columns for columns in customer_activity_df.columns if customer_activity_df[columns].dtype in numeric]:
         stat, p = normaltest(customer_activity_df[columns], nan_policy='omit')
         print('Statistics=%.3f, p=%.3f' % (stat, p))
         customer_activity_df.hist(column=[columns],bins=7,edgecolor='black')
         pyplot.title('Histograms of normality for {}'.format(columns))
         pyplot.show() 
        
        qq_plot = qqplot(customer_activity_df['page_values'] , scale=1 ,line='q')
        pyplot.title('QQ plots of normality for Page Values')
        pyplot.show()

     def impute_nulls(self):
        """Imputation of nulls values.
        key functions:
        .fillna() -- impute the null values on either the mean or median for numeric columns depending on the normality test and mode for the category columns.
        """
        customer_activity_df['administrative'] = customer_activity_df['administrative'].fillna(customer_activity_df['administrative'].mean())
        customer_activity_df['administrative_duration'] = customer_activity_df['administrative_duration'].fillna(customer_activity_df['administrative_duration'].median())
        customer_activity_df['informational_duration'] = customer_activity_df['informational_duration'].fillna(customer_activity_df['informational_duration'].median())
        customer_activity_df['product_related'] = customer_activity_df['product_related'].fillna(customer_activity_df['product_related'].median())
        customer_activity_df['product_related_duration'] = customer_activity_df['product_related_duration'].fillna(customer_activity_df['product_related_duration'].median())
        mode = customer_activity_df['operating_systems'].mode()[0]
        customer_activity_df['operating_systems'] = customer_activity_df['operating_systems'].fillna(mode)
         
     def create_copy_deep(self):
        """Creating a copy of the dataframe(df).
        key functions:
        self.impute_nulls() -- ensure the impute_nulls method remove nulls in the new df.
        .copy(deep=True) -- creates a new copy of the customer_activity_df and ensure index remains the same in the new df.
        return df_copy_deep -- returns the new df.
        """ 
        self.impute_nulls()
        df_copy_deep = customer_activity_df.copy(deep=True)
        return df_copy_deep
       
     def skew_correction(self):
        """Corrected skew data in numeric columns.
        key functions:
        for loop -- Iterates through the columns that are in the numeric variables.
        sns.histplot() -- creates an histogram for each of the columns in the for loop.
        """
        small_constant = 0.001
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        for columns in [columns for columns in customer_activity_df.columns if customer_activity_df[columns].dtype in numerics]:
            customer_activity_df[columns] = np.log(customer_activity_df[columns]+ small_constant)
            skewness_after_log_transform = customer_activity_df[columns].skew()
            print('Skewness after log transform:', skewness_after_log_transform)
            sns.histplot(customer_activity_df[columns], kde=True)
            pyplot.title('Dataset  After Log Transform')
            pyplot.show()
     
     
     def outliers_trimming(self):
         """Winsorization on outliers.
         key functions:
         for loop -- iterates through the columns that are in the columns_to_clean and calculates the IQR, lower and upper bounds.
         np.where -- used the to trim the outliers based on the lower and upper bounds.
         return customer_activity_df -- return the customer_activity_df with the outliers trimmed.
         """
         columns_to_clean = ['informational', 'informational_duration', 'page_values']
         columns_to_clean = ['informational', 'informational_duration', 'page_values']
         for column in columns_to_clean:
            Q1 = customer_activity_df[column].quantile(0.25)
            Q3 = customer_activity_df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            lower_winsor = customer_activity_df[column].quantile(0.05)
            upper_winsor = customer_activity_df[column].quantile(0.95)
            customer_activity_df[column] = np.where(customer_activity_df[column] == lower_bound, lower_winsor, customer_activity_df[column])
            customer_activity_df[column] = np.where(customer_activity_df[column] == upper_bound, upper_winsor, customer_activity_df[column])
         return customer_activity_df
    
     def outliers_removal(self):
         """Removal of outliers.

         key functions:
         for loop -- iterates through the columns that are in the columns_to_clean and calculates the IQR, lower and upper bounds.
         nested for loop -- iterates through the customer_activity_df[columns] and if a data point is less or more then the lower and upper bounds it is removed.
         return customer_activity_df -- returns the customer_activity_df with the outliers removed.
         """
         columns_to_clean = ['product_related', 'exit_rates']
         for column in columns_to_clean:
           Q1 = customer_activity_df[column].quantile(0.25)
           Q3 = customer_activity_df[column].quantile(0.75)
           IQR = Q3 - Q1
           lower_bound = Q1 - 1.5 * IQR
           upper_bound = Q3 + 1.5 * IQR
        
           Outlier = []
           for i in customer_activity_df[column]:
            if i <= lower_bound or i >= upper_bound:
                Outlier.append(i)
                customer_activity_df.drop(customer_activity_df[customer_activity_df[column] == i].index, inplace=True)
     
           Index_Outlier = customer_activity_df[(customer_activity_df[column] <= lower_bound) | (customer_activity_df[column] >= upper_bound)].index
           customer_activity_df.drop(Index_Outlier, inplace=True)
    
         return customer_activity_df

     def product_related_duration_outliers_removal(self):
         """Outliers removal.
         
         key function:
         customer_activity_df['product_related_duration'][customer_activity_df['product_related_duration'] > -5] -- removes outliers that are greter than -5.
         """
         customer_activity_df['product_related_duration'] = customer_activity_df['product_related_duration'][customer_activity_df['product_related_duration'] > -5]
         print(f"Skew of product_related_duration column is {customer_activity_df['product_related_duration'].skew()}")
    
     def collinearity(self):
         """Correlation Matrix.

         key functions:
         .select_dtypes(include=['number']).columns -- includes columns that numeric.
         .corr() -- used to create an correlation between variables.
         sns.heatmap -- plots the correaltion matrix as an heatmap.
         """
         numeric_columns = customer_activity_df.select_dtypes(include=['number']).columns
         customer_activity_df_numeric = customer_activity_df[numeric_columns]
         correlation_matrix = customer_activity_df_numeric.corr()
         pyplot.figure(figsize=(12, 10))
         sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
         pyplot.title('Correlation Heatmap')
         pyplot.show()
     
     def dropped_overcorrealated_variables(self):
         """Overcorrelated variables dropped.

         key functions:
         .drop() -- used to drop all overcorrelated variables. Axis=1 refers to the columns and inplace=True mean that the df will be the same when variables are removed.
         """
         customer_activity_df.drop(['administrative', 'administrative_duration'], axis=1, inplace=True)

class DataInsight():
    """This class is called data_insight
    key functions:
    self.customer_activity_instance = customer_activity_instance -- used to create instance(self) which is used to allow use to use the copied df (df_copied_deep) created in the other class
    """
    def __init__(self, customer_activity_instance):
       self.customer_activity_instance = customer_activity_instance

    
    def weekend_sales(self):
       """Barchart of sales.

       key functions:
       sns.barplot() -- creates a barchart of sales at weekend compared to weekday.
       """
       sns.barplot(data=customer_activity_instance.df_copy_deep, y='revenue', x='weekend')
       pyplot.title('Barchart of Sales at Weekends')
       pyplot.show()     
   
    def revenue_based_on_region(self):
       """Barchart of sales.

       key functions:
       sns.barplot() -- creates a barchart of sales based on region.
       """
       sns.barplot(data=customer_activity_instance.df_copy_deep, y='revenue', x='region')
       pyplot.title('Barchart of Sales based on Region')
       pyplot.xticks(rotation=45)
       pyplot.show()   
     
    def sales_based_on_trafic_type(self):
       """Barchart of sales. 

       key functions:
       sns.barplot() -- creates a barchart of sales based on traffic_type.
       """
       pyplot.figure(figsize=(16,10))
       sns.barplot(data=customer_activity_instance.df_copy_deep, y='revenue', x='traffic_type')
       pyplot.title('Barchart of Sales based on Traffic Type')
       pyplot.xticks(rotation=45)
       pyplot.show()
    
    def sales_based_on_month(self): 
       """Barchart of sales.

       key functions:
       sns.barplot() -- creates a barchart of sales based on month.
       """
       sns.barplot(data=customer_activity_instance.df_copy_deep, y='revenue', x='month')
       pyplot.title('Barchart of Sales based on Month')
       pyplot.show()
    
    def precentage_of_time_of_each_task(self):
       """Pie chart of precentage of time spent on each task. 

       key functions:
       .sum() -- calculates the sum time spent on administrative, product_related, informational tasks. 
       total_duration_all_task -- adds all the times ups to calculate total time. 
       precentage_admin, info, product -- calculates the time as a precentage for the three groups. 
       pyplot.pie() -- used to display the precentages as a pie chart.
       """
       total_duration_admin = customer_activity_instance.df_copy_deep['administrative_duration'].sum() 
       total_duration_info = customer_activity_instance.df_copy_deep['informational_duration'].sum() 
       total_duration_product = customer_activity_instance.df_copy_deep['product_related_duration'].sum()
       total_duration_all_task =  total_duration_admin + total_duration_info + total_duration_product
       precentage_admin = (total_duration_admin / total_duration_all_task)*100
       precentage_info = (total_duration_info / total_duration_all_task)*100
       precentage_product = (total_duration_product / total_duration_all_task)*100
       labels = [ 'Administrative', 'Product_Related', 'Informational']
       sizes = [precentage_admin, precentage_info, precentage_product]
       explode = (0, 0, 0)
       pyplot.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=140)
       pyplot.axis('equal')
       pyplot.title('Percentage of time spent on each task')
       pyplot.show()
    
    def popular_informational_tasks(self):
       """Barchart of popular informational tasks. 

       key functions:
       total_duration.plot() -- creates a barchart of each informational task based on the total_duration variable.
       """  
       task_counts = customer_activity_instance.df_copy_deep['informational'].value_counts()
       total_duration = customer_activity_instance.df_copy_deep.groupby('informational')['informational_duration'].sum()
       total_duration.plot(kind='bar', xlabel='informational', ylabel='Frequency', title='Frequency of Informational tasks')
       pyplot.show()
    
    def popular_administrative_tasks(self):
       """Barchart of popular administrative tasks.

       key functions:
       total_duration.plot() -- creates a barchart of each administrative task based on the total_duration variable.
       """
       task_counts = customer_activity_instance.df_copy_deep['administrative'].value_counts()
       total_duration = customer_activity_instance.df_copy_deep.groupby('administrative')['administrative_duration'].sum()
       total_duration.plot(kind='bar', xlabel='administrative', ylabel='Frequency', title='Frequency of Administrative tasks')
       pyplot.show()
    
    def operating_systems_used(self):
       """Barchart of operating systems used.

       key functions:
       mobile_os, desktop_os, other_os -- splits the operating systems into three groups.
       .isin() -- checks that the following operating systems are in the columns and in the assigned group.
       .value_counts() -- counts the occurance of each of the operating systems in the column.
       pyplot.bar() -- plots the values counts as a barchart.
       """
       mobile_os = ['Android', 'iOS']
       desktop_os = ['Windows', 'MACOS', 'ChromeOS', 'Ubuntu',]
       other_os = ['Other']
       customer_activity_instance.df_copy_deep['category'] = ''
       customer_activity_instance.df_copy_deep.loc[customer_activity_instance.df_copy_deep['operating_systems'].isin(mobile_os), 'category'] = 'Mobile'
       customer_activity_instance.df_copy_deep.loc[customer_activity_instance.df_copy_deep['operating_systems'].isin(desktop_os), 'category'] = 'Desktop'
       customer_activity_instance.df_copy_deep.loc[customer_activity_instance.df_copy_deep['operating_systems'].isin(other_os), 'category'] = 'Other'    
       category_count = customer_activity_instance.df_copy_deep['category'].value_counts()
       pyplot.bar(category_count.index, category_count.values)
       pyplot.xlabel('Category')
       pyplot.ylabel('Count')
       pyplot.title('Count of Operating Systems')
       pyplot.show()
    
    def precent_operating_system_used(self):
       """Pie chart of precentage of time of each operating system used.

       key functions:
       mobile_count, Desktop_count, Other_count  -- counts the occurance of each of the operating systems in the column. 
       total_value_count -- adds all the counts up together to calculate total time.
       precentage_mobile, Desktop, Other -- calculates the time as a precentage for the three groups.
       pyplot.pie() -- used to display the precentages as a pie chart.
       """
       mobile_count = customer_activity_instance.df_copy_deep['category'].value_counts()['Mobile']
       Desktop_count = customer_activity_instance.df_copy_deep['category'].value_counts()['Desktop']
       Other_count = customer_activity_instance.df_copy_deep['category'].value_counts()['Other']
       total_value_count = mobile_count + Desktop_count + Other_count
       precentage_mobile = (mobile_count / total_value_count)*100
       precentage_Desktop = (Desktop_count / total_value_count)*100
       precentage_Other = (Other_count / total_value_count)*100
       labels = [ 'Mobile', 'Desktop', 'Other']
       sizes = [precentage_mobile, precentage_Desktop, precentage_Other]
       explode = (0, 0, 0)
       pyplot.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=140)
       pyplot.axis('equal')
       pyplot.title('Percentage of time spent on each Operating Systems')
       pyplot.show()
    
    def visitor_based_on_operating_systems(self):
       """Barchart of visitors based on operating systems.

       key functions:
       sns.countplot() -- creates a barchart of traffic_type based on operating systems.
       """
       mobile_os = ['Android', 'iOS']
       desktop_os = ['Windows', 'MACOS', 'ChromeOS', 'Ubuntu',]
       sns.countplot(x=customer_activity_instance.df_copy_deep['traffic_type'], hue=customer_activity_instance.df_copy_deep['operating_systems'].apply(lambda x: 'Mobile' if x in mobile_os else 'Desktop'), data=customer_activity_instance.df_copy_deep, palette='muted')
       pyplot.xlabel('Traffic Type')
       pyplot.xticks(rotation=45)
       pyplot.ylabel('Count')
       pyplot.title('Count of Device type by Traffic Type')
       pyplot.show()
    
    def browsers_based_on_operating_systems(self):
       """Barchart of browers based on operating systems.

       key functions:
       sns.countplot() -- creates a barchart of browers based on operating systems.
       """
       mobile_os = ['Android', 'iOS']
       desktop_os = ['Windows', 'MACOS', 'ChromeOS', 'Ubuntu',]
       sns.countplot(x=customer_activity_instance.df_copy_deep['browser'], hue=customer_activity_instance.df_copy_deep['operating_systems'].apply(lambda x: 'Mobile' if x in mobile_os else 'Desktop'), data=customer_activity_instance.df_copy_deep, palette='muted')
       pyplot.xlabel('Browsers')
       pyplot.xticks(rotation=45)
       pyplot.ylabel('Count')
       pyplot.title('Browsers broken down by Device Type')
       pyplot.show()
    
    def revenue_by_traffic_based_on_region(self):
       """Barchart of revenue of traffic_type based on region.

       key functions:
       pivot_table() -- creates pivot table for the data.
       pivot_table.plot() -- creates a stacked barchart of revenue from traffic_type based on region.
       """
       pivot_table = customer_activity_instance.df_copy_deep.pivot_table(index='traffic_type', columns='region', values='revenue',aggfunc='mean')
       pivot_table.plot(kind='bar', stacked=True)
       pyplot.xlabel('Traffic Type')
       pyplot.xticks(rotation=45)
       pyplot.ylabel('revenue')
       pyplot.title('Revenue of Traffic Type based on Region')
       pyplot.legend(title='Region', loc='upper left')
       pyplot.show()
    
    def traffic_bounce_rate(self):
       """Barchart of bounce rate of traffic types.

       key functions:
       sns.barplot() -- creates a barchart of bounce rates for traffic types.
       """
       pyplot.figure(figsize=(16,10))
       sns.barplot(data=customer_activity_instance.df_copy_deep, y='bounce_rates', x='traffic_type')
       pyplot.title('Barchart of Bounce Rates based on traffic type')
       pyplot.xticks(rotation=45)
       pyplot.show()

    def traffic_bounce_rate_based_on_region(self):
       """Barchart of bounce rate of traffic_type based on region.

       key functions:
       pivot_table() -- creates pivot table for the data.
       pivot_table.plot() -- creates a stacked barchart of bounce rates for traffic_type based on region.
       """
       pivot_table = customer_activity_instance.df_copy_deep.pivot_table(index='traffic_type', columns='region', values='bounce_rates',aggfunc='mean')
       pivot_table.plot(kind='bar', stacked=True)
       pyplot.xlabel('Traffic Type')
       pyplot.xticks(rotation=45)
       pyplot.ylabel('Bounce Rates')
       pyplot.title('Bounce Rates of Traffic Type based on Region')
       pyplot.legend(title='Region', loc='upper left')
       pyplot.show()
    
    def months_sales_from_ads(self):
       """Barchart of revenue of month based on traffic_type.

       key functions:
       pivot_table() -- create pivot table for the data.
       pivot_table.plot() -- creates a stacked barchart of revenue from month based on traffic_type.
       """
       num_colours = 20
       custom_palette = sns.color_palette('tab20', num_colours)
       pivot_table = customer_activity_instance.df_copy_deep.pivot_table(index='month', columns='traffic_type', values='revenue',aggfunc='mean')
       pivot_table.plot(kind='bar', stacked=True, color=custom_palette)
       pyplot.xlabel('Month')
       pyplot.xticks(rotation=45)
       pyplot.ylabel('Revenue')
       pyplot.title('Revenue of month based on Traffic Type')
       pyplot.legend(title='Traffic Type', bbox_to_anchor=(1, 0.5, 0.5, 1), loc='upper left')
       pyplot.tight_layout()
       pyplot.show()
    
    def precent_of_customer_type_purchase(self):
       """Precent of purchase rate based on customer type.

       key functions:
       .groupby() -- calculates the purchase rate for customer type.
       purchase_rate.plot -- creates a barchart of purchase rate based on customer type.
       """
       purchase_rate = customer_activity_instance.df_copy_deep.groupby('visitor_type')['revenue'].mean()*100
       pyplot.figure(figsize=(8, 6))
       purchase_rate.plot(kind='bar', color='skyblue')
       pyplot.title('Purchase Rate by Customer Type')
       pyplot.xlabel('Customer Type')
       pyplot.ylabel('Purchase Rate (%)')
       pyplot.grid(axis='y', linestyle='--', alpha=0.7)
       pyplot.tight_layout()
       pyplot.show()
    
    def type_of_traffic_contributing_to_sales(self):
       """Barchart of traffic_type contributing to sales.

       key functions:
       Direct_traffic, Social_traffic, Advertising_traffic  -- splits the traffic type into three groups.
       .isin() -- checks that the following traffic types are in the column and are in the assigned group.
       .value_counts() -- counts the occurance of each of the traffic types in the column.
       pyplot.bar() -- plots the values counts as a barchart. 
       """
       Direct_traffic = ['Google search', 'Bing search', 'Direct Traffic', 'Yahoo Search', 'Yandex search',  'DuckDuckGo search', 'Other' ]
       Social_traffic = ['Facebook ads', 'Instagram ads', 'Youtube ads', 'Twitter', 'Youtube channel', 'Instagram Page', 'Tik Tok ads', 'Facebook page', 'Tik Tok page', 'Pinterest', 'Other']
       Advertising_traffic = ['Affiliate marketing', 'Newsletter', 'Other']
       customer_activity_instance.df_copy_deep['traffic_category'] = ''
       customer_activity_instance.df_copy_deep.loc[customer_activity_instance.df_copy_deep['traffic_type'].isin(Direct_traffic), 'traffic_category'] = 'Direct'
       customer_activity_instance.df_copy_deep.loc[customer_activity_instance.df_copy_deep['traffic_type'].isin(Social_traffic), 'traffic_category'] = 'Social'
       customer_activity_instance.df_copy_deep.loc[customer_activity_instance.df_copy_deep['traffic_type'].isin(Advertising_traffic), 'traffic_category'] = 'Advertising'  
       category_count = customer_activity_instance.df_copy_deep['traffic_category'].value_counts()
       pyplot.figure(figsize=(16,10))
       sns.barplot(data=customer_activity_instance.df_copy_deep, y='revenue', x='traffic_category')
       pyplot.title('Barchart of Sales based on type of Traffic')
       pyplot.xticks(rotation=45)
       pyplot.show()
        
customer_activity_df = pd.read_csv('customer_activity.csv')
test = Plotter(customer_activity_df) 
test_impute = DataFrameTransform(customer_activity_df)
customer_activity_instance = DataFrameTransform(customer_activity_df)
data = DataInsight(customer_activity_instance)
test_impute.null_values()
test_impute.normality_test()
test_impute.impute_nulls()
test.visualise_null_values()
test_impute.create_copy_deep()
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
test_impute.dropped_overcorrealated_variables()
data.weekend_sales()     
data.revenue_based_on_region()     
data.sales_based_on_trafic_type()
data.sales_based_on_month()
data.precentage_of_time_of_each_task()     
data.popular_informational_tasks()
data.popular_administrative_tasks() 
data.operating_systems_used()
data.precent_operating_system_used()
data.visitor_based_on_operating_systems()
data.browsers_based_on_operating_systems()
data.revenue_by_traffic_based_on_region()
data.traffic_bounce_rate()
data.traffic_bounce_rate_based_on_region()
data.months_sales_from_ads()
data.precent_of_customer_type_purchase()
data.type_of_traffic_contributing_to_sales()
help(plotter)
help(DataFrameTransform)
help(data_insight)
