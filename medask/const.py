from os import environ


# API Key for openai. Needed only on the server.
KEY_OPENAI = environ.get("KEY_OPENAI", "")

# API key for anthropic. Needed only for benchmarking.
KEY_ANTHROPIC = environ.get("KEY_ANTHROPIC", "")

# API key for replicate. Needed only for benchmarking.
# Note: replicate API isn't supported yet.
KEY_REPLICATE = environ.get("KEY_REPLICATE", "")

# API key for mistral. Needed only for benchmarking.
KEY_MISTRAL = environ.get("KEY_MISTRAL", "")

# API key for deepseek. Needed only for benchmarking.
KEY_DEEPSEEK = environ.get("KEY_DEEPSEEK", "")
