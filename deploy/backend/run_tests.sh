#!/bin/bash

# Ensure we are in the script's directory
cd "$(dirname "$0")"

# Set python path so pytest can find main.py and config_loader.py
export PYTHONPATH=$(pwd):$PYTHONPATH

# Use pytest to run the tests
python -m pytest tests/ -v
