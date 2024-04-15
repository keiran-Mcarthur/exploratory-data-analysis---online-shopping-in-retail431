
import yaml

def load_yaml(file_path):
        with open(file_path, 'r') as file:
            credentials = yaml.safe_load(file)
            return credentials
   


file_path = "credentials.yaml"
credentials = load_yaml(file_path)
print(credentials)
    

