#!/bin/bash

# Script to sync training logs from remote server
# This will pull logs, checkpoints, and statistics from the remote training

# Remote server details
REMOTE_USER="pengliu"
REMOTE_HOST="10.15.225.64"
REMOTE_PATH="/home/pengliu/Private/my_panns_transfer_to_gtzan/"
LOCAL_PATH="."

echo "🔄 Syncing training logs from remote server..."
echo "Remote: $REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH"
echo "Local:  $LOCAL_PATH"
echo ""

# Create local workspace directories if they don't exist
mkdir -p workspaces/emotion_regression/{logs,statistics,checkpoints,predictions}
mkdir -p workspaces/emotion_feedback/{logs,statistics,checkpoints,predictions}

# First, let's check what's actually on the remote
echo "📋 Checking remote workspace structure..."
ssh "$REMOTE_USER@$REMOTE_HOST" "ls -la $REMOTE_PATH/workspaces/"

echo ""
echo "📋 Checking emotion_regression contents..."
ssh "$REMOTE_USER@$REMOTE_HOST" "ls -la $REMOTE_PATH/workspaces/emotion_regression/ 2>/dev/null || echo 'emotion_regression not found'"

echo ""
echo "📋 Checking emotion_feedback contents..."
ssh "$REMOTE_USER@$REMOTE_HOST" "ls -la $REMOTE_PATH/workspaces/emotion_feedback/ 2>/dev/null || echo 'emotion_feedback not found'"

echo ""
echo "📥 Syncing baseline emotion regression workspace..."
rsync -avz --progress \
    "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/workspaces/emotion_regression/" \
    "./workspaces/emotion_regression/" \
    --exclude="features/" \
    2>/dev/null || echo "Failed to sync emotion_regression"

echo ""
echo "📥 Syncing LRM emotion feedback workspace..."
rsync -avz --progress \
    "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/workspaces/emotion_feedback/" \
    "./workspaces/emotion_feedback/" \
    --exclude="features/" \
    2>/dev/null || echo "Failed to sync emotion_feedback"

echo ""
echo "📥 Syncing any other workspace directories..."
ssh "$REMOTE_USER@$REMOTE_HOST" "find $REMOTE_PATH/workspaces/ -maxdepth 1 -type d -name '*' | grep -v '^$REMOTE_PATH/workspaces/$'" | while read remote_dir; do
    workspace_name=$(basename "$remote_dir")
    if [[ "$workspace_name" != "emotion_regression" && "$workspace_name" != "emotion_feedback" ]]; then
        echo "Found additional workspace: $workspace_name"
        rsync -avz --progress \
            "$REMOTE_USER@$REMOTE_HOST:$remote_dir/" \
            "./workspaces/$workspace_name/" \
            --exclude="features/" \
            2>/dev/null || echo "Failed to sync $workspace_name"
    fi
done

echo ""
echo "✅ Sync completed!"
echo ""
echo "📊 Available logs:"
find workspaces -name "*.log" -type f 2>/dev/null | while read logfile; do
    size=$(ls -lh "$logfile" | awk '{print $5}')
    echo "  - $logfile ($size)"
done

echo ""
echo "📈 Available statistics:"
find workspaces -name "*.pkl" -type f 2>/dev/null | while read statfile; do
    size=$(ls -lh "$statfile" | awk '{print $5}')
    echo "  - $statfile ($size)"
done

echo ""
echo "💾 Available checkpoints:"
find workspaces -name "*.pth" -type f 2>/dev/null | while read checkpoint; do
    size=$(ls -lh "$checkpoint" | awk '{print $5}')
    echo "  - $checkpoint ($size)"
done

echo ""
echo "📁 Available predictions:"
find workspaces -name "*.csv" -type f 2>/dev/null | while read csvfile; do
    size=$(ls -lh "$csvfile" | awk '{print $5}')
    echo "  - $csvfile ($size)"
done 