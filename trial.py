from sqlalchemy import create_engine
from credentials import load_yaml
import pandas as pd 
class RDSDatabaseConnector:
    def __init__(self, credentials: dict) -> None:
        self.credentials = credentials
      

    def create_engine(self):
        #credentials = RDSDatabaseConnector().load_yaml()
        DATABASE_TYPE = 'postgresql'
        DBAPI = 'psycopg2'
        ENDPOINT = self.credentials['RDS_HOST']
        USER = self.credentials['RDS_USER']
        PASSWORD = self.credentials['RDS_PASSWORD']
        PORT = int(self.credentials.get('RDS_PORT',5432))
        DATABASE = self.credentials['RDS_DATABASE']
        engine = create_engine(f"{DATABASE_TYPE}+{DBAPI}://{USER}:{PASSWORD}@{ENDPOINT}:{PORT}/{DATABASE}")
        return engine
    
 

    def customer_activity_data(self):
        
        sql_query = f"SELECT * FROM customer_activity"
        engine = self.create_engine()
        try:
            with engine.connect() as connection:
                df = pd.read_sql_query(sql_query, connection)
                print(df)
                return df
        except Exception as e:
            print(f"Error fetching data from the database: {e}")
    
    
        
    
            

        
    def saved_as_csv(self, df, file_path):
        df.to_csv(file_path, index=False)
        file_path = 'customer_activity.csv'
  

 


if __name__ == "__main__":
    credentials_yaml = load_yaml("credentials.yaml")
    RDSDatabaseConnector(credentials_yaml)


    
   
    
    
    
