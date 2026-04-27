import os
import re

d = "/home/kaushik/Desktop/manish/MLOPS_Project/latent-faults-slipgen/src/models"

for f in os.listdir(d):
    if not f.endswith(".py"): continue
    
    path = os.path.join(d, f)
    with open(path, "r") as file:
        content = file.read()
        
    replacements = {
        "epochs=10": "epochs=config['train']['epochs']",
        "epochs=1": "epochs=config['train']['epochs']",
        "lr=1e-4": "lr=config['train']['learning_rate']",
        "batch_size=16": "batch_size=config['train']['batch_size']",
        "latent_dim=16": "latent_dim=config['model']['latent_dim']",
    }
    
    if "import yaml" not in content:
        content = "import yaml\nwith open('/home/kaushik/Desktop/manish/MLOPS_Project/latent-faults-slipgen/config.yaml', 'r') as f:\n    config = yaml.safe_load(f)\n" + content
        
    for k, v in replacements.items():
        content = content.replace(k, v)
        
    with open(path, "w") as file:
        file.write(content)
