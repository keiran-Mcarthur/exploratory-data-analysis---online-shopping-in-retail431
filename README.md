# Exploratory Data Analysis within Online Shopping in Retail

## Description 
The aim of this project is to complete an exploratory data analysis of online shoppers within the retail domain, so far I have used object oriented programming to create a class that has allowed me to connect to an RDS database and save the following dataset as  csv. Furthermore, I then converted the csv file to a pandas dataframe for some inital analysis whch includes data preping and cleaning the data ready for analysis. Finally, I have created graphs from my dataset which have been used to help me answer question based on our customer needs and demands. 

The aim by the end of this project is to eventaully show my skills as a data analysis and how I can use these skills to make data driven decision. So far, thoughout this project I haved learned so much includig the importance of keeping your database connection details safe especially when working with Amazon web service (AWS), but also the importance of code structure and writing when running and solving issuses with the code. I also learnt the importance of using for loops to reduce reptition of certain code to help improve readability code but also my code structure and length. Futhermore, I have also learned the importance graph selection when it comes to data visualisation and the affect it can have on the readability and interpertation of my finding to others. 

## Useage Instructions 
To use the python files within this repository simply download the files using the dowload button and then open the files in python to run the files. The order to run the files is as followed db_utils.py, csv_to_pandas.py, data_transformation.py, dataframe_info.py and impute_null_values.py. 

# File Structure 
## .gitignore 
This file was created to keep the credentials of the RDS database secure through placing the credentials. yaml file with a .gitignore file it ensures that the credentials are not pushed to github when you upload your files. This file also a csv copy of the customer_ctivity data what was produced after running the db_utils.py file and is the required file that will be used for the exploratory data analysis task. 

## credentials.py 
This file was created to convert the data from the yaml file to a python dictionary through creating a function that improted  and readed the yaml file and return the credentials. 

## db_utils.py 
This file was created to connect to the RDS database through the use of object oriented programming. Fristly, I imported pandas, create_engine from SQLalchemy and credentials from the credentials.py. Then I created a class called RDSDatabaseConnector with an method which contained the instance of the class (self) with credentials as a parameter. This was follwed by another method called create_engine which contain the credentials to connect to the RDS database using self, followed by a varaibale called engine which contain the create_engine string which is need to establish a connection to the RDS database (fig1). After this method I created another method called customer_activity_data which used sql_query to select the dataset(customer_activity) that is goig to be used for the project. Once I selected the correct table I used pandas dataframe to extract the data. Finally, created another function called saved_as_csv which was used to save the select dataset as a csv file on my local computer. I than ran the code to produce the required output. 

## csv_to_pandas.py 
This file was created to convert the customer_activity.csv file into a pandas dataframe (customer_activity_df) for intial analysis which include finding the shape of the dataset and null values each column has. This was achieve through importing pandas then creating a function which read the csv as a pandas dataframe and printed out the shape and the infomation about the dataframe (fig2).

## Data_Transformation.py 
This file was created to convert any columns from the customer_activity_df to another data type. The file includes one import (pandas) followled by a class called DataTransfrom with an method which contained the instance of the class (self) with customer_activity_df as a parameter. This is followed by individual methods for coverting the columns which are an object data type to an category data type. 

## Dataframe_info.py 
This was created as part of the milestone 3 task. Within this python file contains import modules (pandas, numpy, matplotlib) followed by a class called DataFrameInfo with an method which contained the instance of the class (self) with customer_activity_df as a parameter. This is followed by the following methods (describe, statistics_extraction, category_distinct_values and null_values). The describe method contains the python functions describe(), .info(), .shape on the customer_activity_df. The next method is the statistics_extraction methods which contains the functions .mean(), .median(), .std() to calculate the mean, median and standard devaition for the numeric data type columns in the customer_activity_df (rounded to one decimal place). The next method is the category_distinct_values method which contains a for loop that iterates through the customer_activity_df and for any column that is ethier a object or category and it prints the value counts of those columns. Finally, the final method is the null_values method which contains the calculation for the null_count with the dataset using the .isna().sum() functions. This is followed by the calculation for the precentage of nulls for each column (rounded to one decimal place) using the function .isna().mean()*100.

## Impute_Null_Values.py 
This python file was also created as part of the milestone 3 task and contains two classes (plotter and DataFrameTransfom). The file contains a number of imports (pandas, scipy.stats, statsmodels.graphics.gofplots, matplotlib, numpy, seaborn and plotly.express). Futhermore, this file also contains a class called Data_Insights whiich includes the same imports and was created for the Milestone 4 task. 

### Plotter
This class is called plotter and has a method which contains the instance of the class (self) with customer_activity_df as a parameter. This is followed by the following methods(visualise_null_values, skewed_data, outliers_detection_visual, outliers_dection_statisiacal and frequncy_tables). The first method is the visualise_null_values which contains the calculation of the nulls in the dataset followed by a barchart showing showing the number of null after removal. This was created as visualisation tool to check that there were no null values left in the dataset after removal. The next method is the skewed_data method which contains the code for an histogram using the .hist() function to create a histogram of selected column in the dataset to visually check the skewness of the data. This followed by the .skew() which prints the value of the skew for the selected column. This method also contains a QQ plot for a selected column using the qq_plot() function and pyplot.show() to visualise the QQ plot. This was used in hand with the histogram to determined the skewness of a column. The next method outliers_detection_visual method which contains visualised way to detect outliers in the dataset using histograms and box plot. The histogram was created using a for loop to iterate through the numeric data type columns in the dataset using the .hist() function(fig 3). The box plot was created using the same technique but with .boxplot() function (fig 3). These graphs were used to help visually determine outliers in each column. The next method is the outliers_detection_statisical method which was used to calculate the Inter Quatertile Range (IQR) for each variable (fig 4). The methods outliers_detection_visualise and outliers_detection_statistical were used together to determine outliers in each column. The final method is the frequncy table method which is used to visualise the category data type columns to examine if there are any outliers within the category columns. 

### DataFrameTransform
This class is called DataFrameTransform and has a method which contains the instance of the class (self) with customer_activity_df as a parameter. This is followed by the following methods(null_values, normality_test, impute_nulls, create_copy_deep, skew_correction, outliers_trimming, outliers_removal, product_related_duration_outliers_removal, collinearity and dropped_overcorrealated_variables). The first method is the null_values method which contains the calculation for the count of null values for each column and the precentage of null in each column. The next method is the normality_test method which contains the function normaltest() to calculate the p value for each column in dataset to test to see if the columns were normally distributed. As well as the p value a histogram and QQ plot were also used to visually visualise the distribution of the data within each column. The next method is the impute_nulls method which contains the fillna function which was used to impute the null values using the mean and median based on the normal distribution of the data for numeric data types and the mode was used for categorical data type columns. The next method is the create_deep_copy method which was used to create a copy of the customer_activity_df after the null values had been imputed but before the skew correction and outliers removal so the values where not altered by the skew tranformation. The next method is the skew_correction method which contains for loop that iterates through the numeric columns of the dataset and performs a log transformation using the np.log() function followed by a histogram to visualise the change (fig 5). This log transformantion was done to try and make the variable more normally distributed. The next method is the outliers_trimming method which contained a winsorization calculation on selected columns using a for loop to trim outliers to reduce there effcets on the data (fig 6). This method was choosen based on trial and error between other ways of dealing with outliers. The next column is the outliers_removal method which contains two for loops to remove outliers from the selected columns. The first for loop iterates through the selected columns and calculates the IQR and the lower and upper bounds (fig 7). The next for loop is a nested for loop which iterates through the columns data and removes any data points that are less than the lower_bound and greater than the upper_bound (fig 7). The next column is the product_related_duration_outliers_removal method which contains the formula to remove any data points that are greater than minus 5 from the datasets. The reason for this is that even through there are still outliers after removing data points that are greater than minus 5 the rest of the points are gathered near the end of the lower and upper bounds and after researching and visulaising the histogram and the box plot I decided to leave the new outliers in the dataset (fig 8). The next method is the collinearity method which contains correalation matrix for the selected columns in the dataset using the .corr() and the sns.heatmap() function (fig 9). The result of this was a heatmap of correalations of each variable and after viewing and analysing the correalations I decided to remove two varaiables from the dataset. The final method is the dropped_overcorrealated_variables method which contain dropna() function which removes the administrative and administrative_duration from the customer_activity_df as they were higherly correalated with each other after viewing the correalation matrix. 

### Data_Insight
This Class is called Data_insights and has method which contains the instance of the class (self) with customer_activity_df as a parameter. This is followed by the following methods(weekend_sales, revenue_based_on_region, sales_based_on_trafic_type, sales_based_on_month, precentage_of_time_of_each_task, popular_informational_tasks, popular_administrative_tasks, operating_systems_used, precent_operating_system_used, visitor_based_operating_systems, browsers_based_on_operating_systems, revenue_by_traffic_based_on_region, traffic_bounce_rate, months_sales_from_ads, precent_of_customer_type_purchase and type_of_traffic_contributing_to_sales) which contain graphs that are required for the Milestone 4 task. These methods contain a mixture of graphs and charts that best visualise the anwser to each of the question in Mileston 4. Example of some of these graphs are the revenue_based_on_month method contains a barchart of revenue based on region (fig 10), precentage_of_time_of_each_task which contains a pie chart of the precentage of time spent on administrative, product_related and informational tasks(fig 11) and finally revenue_by_traffic_based_on_region method which contains a barchart of revenue from traffic_type based region (fig 12).

# Appendics 
## fig 1 - Engine connection code
![fig1](https://github.com/keiran-Mcarthur/exploratory-data-analysis---online-shopping-in-retail431/assets/159048029/2252bd5a-60c8-4eb5-804d-cb54754385d4)

## fig 2 - Converting csv to pandas df 
![fig 2](https://github.com/keiran-Mcarthur/exploratory-data-analysis---online-shopping-in-retail431/assets/159048029/4ff685e0-ea61-4809-b790-4425cd1e8d30)


## fig 3 - outliers_detection_visual
![fig 3](https://github.com/keiran-Mcarthur/exploratory-data-analysis---online-shopping-in-retail431/assets/159048029/8ee68441-48f9-4996-adad-9b385c1f57e2)

## fig 4 -  outliers_detection_statisical
![fig 4](https://github.com/keiran-Mcarthur/exploratory-data-analysis---online-shopping-in-retail431/assets/159048029/1190d039-ec13-41aa-bdfc-bb0a72813b3b)

## fig 5 - skew_correction
![fig 5](https://github.com/keiran-Mcarthur/exploratory-data-analysis---online-shopping-in-retail431/assets/159048029/2c9f740e-bcf5-493e-8f9f-37b59b3b7540)

## fig 6 - outliers_trimming
![fig 6](https://github.com/keiran-Mcarthur/exploratory-data-analysis---online-shopping-in-retail431/assets/159048029/5ddcb8d0-0764-4e37-9611-7ce85ce3eead)

## fig 7 - outliers_removal 
![fig 7 pt1](https://github.com/keiran-Mcarthur/exploratory-data-analysis---online-shopping-in-retail431/assets/159048029/66176837-afcb-490f-a85f-9ef896e564ad)
![fig 7 pt2](https://github.com/keiran-Mcarthur/exploratory-data-analysis---online-shopping-in-retail431/assets/159048029/d241d4d8-a303-49d4-9c17-323deaf82dd7)

## fig 8 - product_related_duration box plot 
![fig pro_re_dur](https://github.com/keiran-Mcarthur/exploratory-data-analysis---online-shopping-in-retail431/assets/159048029/27e6301a-da72-41c2-bfed-42d5605bfa95)

## fig 9 - Correleation Matrix

![fig 9](https://github.com/keiran-Mcarthur/exploratory-data-analysis---online-shopping-in-retail431/assets/159048029/fa127e1d-ea9f-4672-9283-52ca08596927)

## fig 10 - barchart revenue based on month
![fig 10](https://github.com/keiran-Mcarthur/exploratory-data-analysis---online-shopping-in-retail431/assets/159048029/7b5eda5a-843e-45ba-b9a4-257ce935e019)


## fig 11 -  precentage of time spent on administrative, product_related and informational
![fig 11](https://github.com/keiran-Mcarthur/exploratory-data-analysis---online-shopping-in-retail431/assets/159048029/5a60f56e-ca75-4666-a854-bbb646337dd3)


## fig 12 - barchart of revenue from traffic_type based region
![fig 12](https://github.com/keiran-Mcarthur/exploratory-data-analysis---online-shopping-in-retail431/assets/159048029/758318eb-edf9-4b0d-b07d-fcc8c9b84a56)



