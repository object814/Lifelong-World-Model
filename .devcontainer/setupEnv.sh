# By Markus Baumgartner (https://github.com/applied-ai-lab/Galileo)

#!/bin/bash

# This script helps to build the dev container with proper environment variables
set -e

# Get the current user's UID
CURRENT_UID=$(id -u)

# Get the script directory to ensure we create .env in the right location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"

# Check if .env file exists, create with defaults if it doesn't
if [ ! -f "$ENV_FILE" ]; then
  echo "Creating default .env file at $ENV_FILE..."
  cat > "$ENV_FILE" << EOF
# User configuration
USER_NAME=$(whoami)
USER_UID=$CURRENT_UID

# Weights & Biases API Key
WANDB_API_KEY=
EOF
  echo ".env file created with default values using current username ($(whoami)) and UID ($CURRENT_UID)."
  echo "Please edit the .env file at $ENV_FILE to set your WANDB_API_KEY and adjust other values if needed."
else
  echo ".env file already exists at $ENV_FILE."
fi

# Source the .env file to get environment variables
source "$ENV_FILE"



# Print values (excluding sensitive information)
echo "Container will be configured with:"
echo "USER_NAME: $USER_NAME"
echo "USER_UID: $USER_UID"
echo "WANDB_API_KEY: [configured: $([ -n "$WANDB_API_KEY" ] && echo "yes" || echo "no")]"