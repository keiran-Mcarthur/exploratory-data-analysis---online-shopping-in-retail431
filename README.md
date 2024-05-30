# Exploratory Data Analysis within Online Shopping in Retail

## Description 
The aim of this project is to complete an exploratory data analysis of online shoppers within the retail domain, so far I have used object oriented programming to create a class that has allowed me to connect to an RDS database and save the following dataset as  csv. Furthermore, I then converted the csv file to a pandas dataframe for some inital analysis whch includes data preping and cleaning the data ready for analysis. Finally, I have created graphs from my dataset which have been used to help me answer question based on our customer needs and demands. 

The aim by the end of this project is to eventaully show my skills as a data analysis and how I can use these skills to make data driven decision. So far, thoughout this project I haved learned so much includig the importance of keeping your database connection details safe especially when working with Amazon web service (AWS), but also the importance of code structure and writing when running and solving issuses with the code. I also learnt the importance of using for loops to reduce reptition of certain code to help improve readability code but also my code structure and length. Futhermore, I have also learned the importance graph selection when it comes to data visualisation and the affect it can have on the readability and interpertation of my finding to others. 

## Useage Instructions 
To use the python files within this repository simply download the files using the dowload button and then open the files in python to run the files. The order to run the files is as followed db_utils.py, csv_to_pandas.py, data_transformation.py, dataframe_info.py and impute_null_values.py. 

# File Structure 
For a more in depth description of this project please the associated wiki
## .gitignore 
This file was created to keep the credentials of the RDS database secure through placing the credentials. yaml file with a .gitignore file it ensures that the credentials are not pushed to github when you upload your files. This file also a csv copy of the customer_ctivity data what was produced after running the db_utils.py file and is the required file that will be used for the exploratory data analysis task. 

## credentials.py 
This file was created to convert the data from the yaml file to a python dictionary through creating a function that improted  and readed the yaml file and return the credentials. 

## db_utils.py 
This file was created to connect to the RDS database through the use of object oriented programming. Fristly, I imported pandas, create_engine from SQLalchemy and credentials from the credentials.py. Then I created a class called RDSDatabaseConnector with an method which contained the instance of the class (self) with credentials as a parameter. This was follwed by another method called create_engine which contain the credentials to connect to the RDS database. After this method I created another method called customer_activity_data which used sql_query to select the dataset that is goig to be used for the project. Once I selected the correct table I used pandas dataframe to extract the data. I then  created another function called saved_as_csv which was used to save the select dataset as a csv file on my local computer. Finally, I used a function called csv_to_pd_df convert the csv to a pandas dataframe. 

## Data_Transformation.py 
This file was created to convert any columns from the customer_activity_df to another data type. The file includes one import (pandas) followled by a class called DataTransfrom with an method which contained the instance of the class (self) with customer_activity_df as a parameter. This is followed by individual methods for coverting the columns which are an object data type to an category data type. 

## Dataframe_info.py 
This was created as part of the milestone 3 task. Within this python file contains import modules (pandas, numpy, matplotlib) followed by a class called DataFrameInfo with an method which contained the instance of the class (self) with customer_activity_df as a parameter. This is followed by the following methods (describe, statistics_extraction, category_distinct_values and null_values). The describe method contains the python functions describe(), info() and shape() on the customer_activity_df. The next method is the statistics_extraction methods which calculate the mean, median and standard devaition for the numeric data type columns in the customer_activity_df. The next method is the category_distinct_values which prints the value counts of any columns in the customer_activity_df that are an object or a category data type. Finally, the final method is the null_values method which contains the calculation for the null_count within the dataset. This is followed by the calculation for the precentage of nulls for each column.

## Impute_Null_Values.py 
This python file was also created as part of the milestone 3 task and contains two classes (Plotter and DataFrameTransfom). The file contains a number of imports (pandas, scipy.stats, statsmodels.graphics.gofplots, matplotlib, numpy, seaborn and plotly.express). Futhermore, this file also contains a class called DataInsights whiich includes the same imports and was created for the Milestone 4 task. 

### Plotter
This class is called plotter and has a method which contains the instance of the class (self) with customer_activity_df as a parameter. This is followed by the following methods(visualise_null_values, skewed_data, outliers_detection_visual, outliers_dection_statisiacal and frequncy_tables). The first method is the visualise_null_values which contains a visualisation showing the number of null after removal. The next method is the skewed_data method which contains the code for histograms of selected columns in the dataset to visually check the skewness of the data. This method also contains a QQ plot for a selected columns. This was used in hand with the histogram to determined the skewness of a column. The next method outliers_detection_visual method which contains visualised way to detect outliers in the dataset using histograms and box plot. These graphs were used to help visually determine outliers in each column. The next method is the outliers_detection_statisical method which was used to calculate the Inter Quatertile Range (IQR) for each variable. The method outliers_detection_visualise and outliers_detection_statistical were used together to determine outliers in each column. The final method is the frequncy_tables method which is used to visualise the category data type columns to examine if there are any outliers within the category columns. 

### DataFrameTransform
This class is called DataFrameTransform and has a method which contains the instance of the class (self) with customer_activity_df as a parameter. This is followed by the following methods(null_values, normality_test, impute_nulls, create_copy_deep, skew_correction, outliers_trimming, outliers_removal, product_related_duration_outliers_removal, collinearity and dropped_overcorrelated_variables). The first method is the null_values method which contains the calculation for the count of null values for each column and the precentage of null in each column. The next method is the normality_test method which determines if the columns were normally distributed. The next method is the impute_nulls method which imputes the null values in the customer_activity_df. The next method is the create_deep_copy method which was used to create a copy of the customer_activity_df. The next method is the skew_correction method which corrects the skew of the numeric columns in he customer_activity_df. The next method is the outliers_trimming method which trims the outliers in the selected columns. The next column is the outliers_removal method which which removes the outliers in the selected columns. The next column is the product_related_duration_outliers_removal method which cntains the formula to remove any data points that are greater than minus 5 from the datasets. The next method is the collinearity method which contains correalation matrix for the selected columns in the dataset. The final method is the dropped_overcorrelated_variables method which  removes the administrative and administrative_duration from the customer_activity_df as they were higherly correalated with each other after viewing the correalation matrix. 

### DataInsight
This Class is called Data_insights and has method which contains the instance of the class (self) with customer_activity_df as a parameter. This is followed by the following methods(weekend_sales, revenue_based_on_region, sales_based_on_trafic_type, sales_based_on_month, precentage_of_time_of_each_task, popular_informational_tasks, popular_administrative_tasks, operating_systems_used, precent_operating_system_used, visitor_based_operating_systems, browsers_based_on_operating_systems, revenue_by_traffic_based_on_region, traffic_bounce_rate, months_sales_from_ads, precent_of_customer_type_purchase and type_of_traffic_contributing_to_sales) which contain graphs that are required for the Milestone 4 task. These methods contain a mixture of graphs and charts that best visualise the anwser to each of the question in Mileston 4.

