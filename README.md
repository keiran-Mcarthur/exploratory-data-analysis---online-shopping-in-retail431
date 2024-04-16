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
This file was created to convert the customer_activity.csv file into a pandas dataframe for intial analysis which include finding the shape of the dataset and null values each column has. This was achieve through importing pandas then creating a function which read the csv as a pandas dataframe and printed out the shape and the infomation about the dataframe. 


