import yaml

def load_api_key(path: str = "api_key.txt") -> str:
    """Read OpenAI API key from a text file."""
    with open(path, "r") as f:
        return f.read().strip()

def load_schema_map(path: str = "mappings/schema_map.yaml") -> dict:
    """Load the schema mapping configuration."""
    with open(path, "r") as f:
        return yaml.safe_load(f)

OPENAI_API_KEY = load_api_key()
SCHEMA_MAP = load_schema_map()
