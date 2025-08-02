import os, yaml

def load_api_key():
    key = os.getenv("OPENAI_API_KEY")
    if key:
        return key.strip()
    with open("api_key.txt", "r") as f:
        return f.read().strip()

def load_schema_map():
    with open("mappings/schema_map.yaml", "r") as f:
        return yaml.safe_load(f)

OPENAI_API_KEY = load_api_key()
SCHEMA_MAP = load_schema_map()["well_logs"]
