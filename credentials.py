
import yaml

def load_yaml():
    with open('credentials.yaml', 'r') as file:
        data = yaml.safe_load(file)
        print(data)
        return data
    

