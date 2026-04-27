import glob

for filepath in glob.glob('/home/kaushik/Desktop/manish/MLOPS_Project/latent-faults-slipgen/src/models/*.py'):
    with open(filepath, 'r') as f:
        content = f.read()

    content = content.replace('pth")))', 'pth"))')
    content = content.replace('",)', '")')

    with open(filepath, 'w') as f:
        f.write(content)
