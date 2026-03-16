import os
import sys

# Get current script path
print(f"Current working directory: {os.getcwd()}")
print(f"Internal __file__ path: {os.path.abspath(__file__)}")

# Try to import config and check its path
try:
    import config
    print(f"Config module loaded from: {os.path.abspath(config.__file__)}")
    print(f"Config.NEO4J_URI: {config.NEO4J_URI}")
except Exception as e:
    print(f"Error loading config: {e}")

# Try to import MedicalGraph and check its path
try:
    from build_medicalgraph import MedicalGraph
    import build_medicalgraph
    print(f"MedicalGraph loaded from: {os.path.abspath(build_medicalgraph.__file__)}")
except Exception as e:
    print(f"Error loading MedicalGraph: {e}")

print("\n--- System Path ---")
for p in sys.path:
    print(p)
