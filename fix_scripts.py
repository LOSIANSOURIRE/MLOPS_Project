import os
import yaml

config_path = "/home/kaushik/Desktop/manish/MLOPS_Project/latent-faults-slipgen/config.yaml"
script_dir = "/home/kaushik/Desktop/manish/MLOPS_Project/latent-faults-slipgen/src/models"

for file in os.listdir(script_dir):
    if not file.endswith('.py'):
        continue
    filepath = os.path.join(script_dir, file)
    with open(filepath, 'r') as f:
        content = f.read()
    
    # We can inject a config reader or replace hyper parameters regex
    # Actually doing this by hand using a quick script is robust
