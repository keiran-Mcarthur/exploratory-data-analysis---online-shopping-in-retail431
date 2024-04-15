import pandas as pd 
from sqlalchemy import create_engine
import psycopg2
class RDSDatabaseConnector:
    def __init__(self, credentials):
        self.username = credentials.get('RDS_user')
        self.password = credentials.get('RDS_password')
        self.host = credentials.get('RDS_host')
        self.port = int(credentials.get('RDS_port',5432))
        self.database = credentials.get('RDS_database')

    def create_engine(self):
        #credentials = RDSDatabaseConnector().load_yaml()
        DATABASE_TYPE = 'postgresql'
        DBAPI = 'psycopg2'
        ENDPOINT = self.credentials['RDS_HOST']
        USER = self.credentials['RDS_USER']
        PASSWORD = self.credentials['RDS_PASSWORD']
        PORT = '5432'
        DATABASE = self.credentials['RDS_DATABASE']
        engine = engine_create(f"{DATABASE_TYPE}+{DBAPI}://{USER}:{PASSWORD}@{EDPOINT}:{PORT}/{DATABASE}")
        return engine
    
   

    def customer_activity_data_extraction(self):
        
        sql_query = f"SELECT * FROM customer_activity"
        try:
            with self.engine.connect() as connection:
                df = pd.read_sql_query(sql_query, connection)
                return df
        except Exception as e:
            print(f"Error fetching data from the database: {e}")
            

        
    def saved_as_csv(self, df, file_path):
        df.to_csv(file_path, index=False)
    

credentials = { 
    'username': 'RDS_user',
    'password': 'RDS_password',
    'host': 'RDS_host',
    'port': 'RDS_port',
    'database': 'RDS_database'
}

db_connector = RDSDatabaseConnector().load_yaml(credentials)
db_connector.create_engine() 

customer_activity_data = db_connector.fetch_customer_activity_data()

file_path = 'customer_activity.csv'
saved_as_csv(customer_activity_data, file_path)