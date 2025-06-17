#!/bin/bash

# Script to check what's actually happening on the remote server
REMOTE_USER="pengliu"
REMOTE_HOST="10.15.225.64"
REMOTE_PATH="/home/pengliu/Private/my_panns_transfer_to_gtzan/"

echo "🔍 Checking Remote Server Status"
echo "================================"
echo "Remote: $REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH"
echo ""

echo "📁 Checking workspace directories..."
ssh "$REMOTE_USER@$REMOTE_HOST" "ls -la $REMOTE_PATH/pytorch/workspaces/ 2>/dev/null || echo 'No workspaces directory found'"

echo ""
echo "🔄 Checking for running Python processes..."
ssh "$REMOTE_USER@$REMOTE_HOST" "ps aux | grep python | grep -v grep || echo 'No Python processes running'"

echo ""
echo "📄 Checking for any log files..."
ssh "$REMOTE_USER@$REMOTE_HOST" "find $REMOTE_PATH -name '*.log' -type f 2>/dev/null || echo 'No log files found'"

echo ""
echo "💾 Checking for any checkpoint files..."
ssh "$REMOTE_USER@$REMOTE_HOST" "find $REMOTE_PATH -name '*.pth' -type f 2>/dev/null || echo 'No checkpoint files found'"

echo ""
echo "📊 Checking recent file modifications..."
ssh "$REMOTE_USER@$REMOTE_HOST" "find $REMOTE_PATH -type f -mmin -60 2>/dev/null | head -10 || echo 'No recent file changes'"

echo ""
echo "🖥️  Checking system resources..."
ssh "$REMOTE_USER@$REMOTE_HOST" "echo 'GPU Status:' && nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || echo 'No GPU info available'"

echo ""
echo "📋 Checking for screen/tmux sessions..."
ssh "$REMOTE_USER@$REMOTE_HOST" "screen -list 2>/dev/null || echo 'No screen sessions'; tmux list-sessions 2>/dev/null || echo 'No tmux sessions'" 