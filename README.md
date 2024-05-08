# Exploratory Data Analysis within Online Shopping in Retail

## Description 
The aim of this project is to complete an exploratory data analysis of online shoppers within the retail domain, so I have used object oriented programming to create a class that has allowed me to connect to an RDS database and save the following dataset as  csv. Furthermore, I then converted the csv file to a pandas dataframe for some inital analysis. The aim by the end of this project is to eventaully show my skills as a data analysis and how I can use these skills to make data driven decision. So far, thoughout this project I haved learned so much includig the importance of keeping your database connection details safe especially when working with Amazon web service (AWS), but also the importance of code structure and writing when running and solving issuses with the code. 

# File Structure 
## .gitignore and New Text Document.txt 
These files were created to keep the credentials of the RDS database secure through placing the credentials. yaml file with a .gitignore file it ensures that the credentials are not pushed to github when you upload your files. The .txt was created to write the path of the credentials.yaml so the .gitignore knew where to idnentify the file. 

## credentials.py 
This file was created to convert the data from the yaml file to a python dictionary through creating a function that improted  and readed the yaml file and return the credentials. 

## db_utils.py 
This file was created to connect to the RDS database through the use of object oriented programming. Fristly, I imported pandas, create_engine from SQLalchemy and credentials from the credentials.py. Then I created a class called RDSDatabaseConnector with an method which contained the instance of the class (self) with credentials as a parameter. This was follwed by another method called create_engine which contain the credentials to connect to the RDS database using self, followed by a varaibale called engine which contain the create_engine string which is need to establish a connection to the RDS database. After this method I created another method called customer_activity_data which used sql_query to select the dataset(customer_activity) that is goig to be used for the project. Once I selected the correct table I used pandas dataframe to extract the data. Finally, created another function called saved_as_csv which was used to save the select dataset as a csv file on my local computer. I than ran the code to produce the required output. 

## customer_activity.csv 
This file is what was produced after running the db_utils.py file and is the required file that will be used for the exploratory data analysis task. 

## csv_to_pandas.py 
This file was created to convert the customer_activity.csv file into a pandas dataframe (customer_activity_df) for intial analysis which include finding the shape of the dataset and null values each column has. This was achieve through importing pandas then creating a function which read the csv as a pandas dataframe and printed out the shape and the infomation about the dataframe.

## Data_Transformation.py 
This file was created to convert any columns from the customer_activity_df to another data type. The file includes one import (pandas) followled by a class called DataTransfrom with an method which contained the instance of the class (self) with customer_activity_df as a parameter. This is followed by individual methods for coverting the columns which are an object data type to an category data type. 

## Dataframe_info.py 
This was created as part of the milestone 3 task. Within this python file contains an import (pandas, numpy, matplotlib) followed by a class called DataFrameInfo with an method which contained the instance of the class (self) with customer_activity_df as a parameter. This is followed by the following methods (describe, statistics_extraction, category_distinct_values and null_values). The describe method contains the python functions describe, .info(), .shape on the customer_activity_df. The next method is the statistics_extraction methods which contains the functions .mean(), .median(), .std() to calculate the mean, median and standard devaition for the numeric data type columns in the customer_activity_df (rounded to one decimal place). The next method is the category_distinct_values method which contains a for loop that iterates through the customer_activity_df and for any column that is ethier a object or category it prints the value counts of those columns. Finally, the final method is the null_values method which contains the calculation for the null_count with the dataset using the .isna().sum() functions. This is followed by the calculation for the precentage of nulls for each column (rounded to one decimal place) using the function .isna().mean()*100.

## Impute_Null_Values.py 
This python file was also created as part of the milestone 3 task and contains two classs which (plotter and DataFrameTransfom). The file contains a number of imports(pandas, scipy.stats, statsmodels.graphics.gofplots, matplotlib, numpy, seaborn and plotly.express)

### Plotter
This class is called plotter and has method which contained the instance of the class (self) with customer_activity_df as a parameter. This is followed by the following methods(visualise_null_values, skewed_data, outliers_detection_visual, outliers_dection_statisiacal and frequncy_tables. The first method is the visualise_null_values which contains the calculation of the nulls in the dataset followed by a barchart showing showing the number of null after removal. This was created as visualisation tool to check that there were no null values left in the dataset after removal. The next method is the skewed_data method which contains the code for an histogram using the .hist() function to create a histogram of selected column in the dataset to visually check the skewness of the data. This followed by the .skew() which prints the value of the skew for the selected column. This method also contains a QQ plot for a selected column using the qq_plot() function and pyplot.show() to visualise the QQ plot. This was hand with the histogram to determined the skewness of a column. The next method outliers_detection_visual method which cntains visualise way to detect outliers in the dataset using histograms and box plot. The histogram was created using a for loop to iterate through the numeric data type columns in the dataset using the .hist() function(fig 1). The box plot was created using the same technique but with .boxplot() function(fig 1). These graphs were used to help visually determine outliers in each column. The next method is the outliers_detection_statisical method which was used to calculate the Inter Quatertile Range (IQR) for each variable(fig 2). The methods outliers_detection_visualise and outliers_detection_statistical were used together to determine outliers in each column. The final method is the frequncy table method which is used to visualise the category data type columns to examine if there are any outliers within the category columns. 

### DataFrameTransform
This class is called DataFrameTransform and has method which contained the instance of the class (self) with customer_activity_df as a parameter. This is followed by the following methods(null_values, normality_test, impute_nulls, skew_correction, outliers_trimming, outliers_removal, product_related_duration_outliers_removal and collinearity). The first method is the null_values method which contains the calculation for the count of null values for each column and the precentage of null in each column. 
