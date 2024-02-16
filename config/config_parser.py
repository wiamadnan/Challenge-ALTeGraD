import argparse
import yaml

def load_yaml_config(config_file_path):
    with open(config_file_path, 'r') as f:
        return yaml.safe_load(f)

def parse_args():
    parser = argparse.ArgumentParser(description='Load configuration from YAML file and/or command line arguments.')
    parser.add_argument('--load_config', required=True, help='Path to the configuration YAML file.')
    args = parser.parse_args()
    
    # Load YAML config
    yaml_config = load_yaml_config(args.load_config)

    # Print the loaded YAML configuration
    print("Loaded YAML configuration:", yaml_config)
    
    return yaml_config  

if __name__ == "__main__":
    config = parse_args()
    print(config) 
