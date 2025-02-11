import os
import sys

# Get the absolute path of the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the BLIP directory to Python path
blip_path = os.path.join(current_dir, 'BLIP')
sys.path.append(blip_path)
