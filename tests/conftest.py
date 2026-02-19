import sys
import os

# Add project root to sys.path so tests can import root-level modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
