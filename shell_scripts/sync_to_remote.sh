#!/bin/bash

# Replace these variables with your specific values
REMOTE_USER="pengliu"
REMOTE_HOST="10.15.225.64"
REMOTE_PATH="/home/pengliu/Private/my_panns_transfer_to_gtzan/"
LOCAL_PATH="."

# Exclude patterns - add more as needed
EXCLUDES=(
    ".git"
    "__pycache__"
    "*.pyc"
    ".DS_Store"
    "venv"
    "*.egg-info"
    "workspaces"
    "*.log"
    "datasets"
)

# Build exclude arguments
EXCLUDE_ARGS=""
for pattern in "${EXCLUDES[@]}"; do
    EXCLUDE_ARGS="$EXCLUDE_ARGS --exclude='$pattern'"
done

# Sync command
eval rsync -avz --progress $EXCLUDE_ARGS \
    "$LOCAL_PATH/" \
    "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/" 