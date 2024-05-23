import pandas as pd

class DataTransform():
   def __init__(self,customer_activity_df):
       self. customer_activity_df = customer_activity_df

   
   def month(self):
       for month in ['Feb', 'Mar', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']:
        customer_activity_df['month'] = customer_activity_df['month'].astype('category')
    
   def region(self):
        for region in ['North America', 'Western Europe', 'Eastern Europe', 'Asia', 'South America', 'Africa', 'Northern Africa', 'Southern Africa', 'Oceania']:
         customer_activity_df['region'] = customer_activity_df['region'].astype('category')
    
   def visitor_type(self):
        for visitor_type in ['Returning_Vistor', 'New_Vistor', 'other']:
          customer_activity_df['visitor_type'] = customer_activity_df['visitor_type'].astype('category')
    
   def browser(self):
        for browser in ['Google Chrome ', 'Safari', 'Mozilla Firefox', 'Microsoft Edge', 'Internet Explorer', 'Samsung Internet', 'Opera', 'Android', 'QQ', 'Sogou Explorer', 'Yandex', 'UC Browser', 'Undetermined']:
          customer_activity_df['browser'] = customer_activity_df['browser'].astype('category')
   
   def traffic_type(self):
        for traffic_type in ['Google search', 'Facebook ads', 'Instagram ads', 'Bing search', 'Youtube ads', 'Twitter', 'Affiliate marketing', 'Youtube channel', 'Instagram Page', 'Direct Traffic', 'Tik Tok ads', 'Yahoo Search', 'Facebook page', 'Yandex search', 'Newsletter', 'Tik Tok page', 'DuckDuckGo search', 'Other', 'Pinterest']:
          customer_activity_df['traffic_type'] = customer_activity_df['traffic_type'].astype('category')
   
   def operating_systems(self):
        for operating_systems in ['Windows', 'MACOS', 'Android', 'iOS', 'ChromeOS', 'Ubuntu', 'Other']:
          customer_activity_df['operating_systems'] = customer_activity_df['operating_systems'].astype('category')
   
   def info(self):
       print(customer_activity_df.info())

customer_activity_df = pd.read_csv('customer_activity.csv')
test = DataTransform(customer_activity_df)
test.month()
test.region()
test.visitor_type()
test.browser()
test.traffic_type()
test.operating_systems()
test.info()

     
