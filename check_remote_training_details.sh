#!/bin/bash

# Script to get detailed information about the training process
REMOTE_USER="pengliu"
REMOTE_HOST="10.15.225.64"
REMOTE_PATH="/home/pengliu/Private/my_panns_transfer_to_gtzan/"

echo "🔍 Detailed Remote Training Check"
echo "================================="
echo ""

echo "📂 Checking emotion_regression workspace contents..."
ssh "$REMOTE_USER@$REMOTE_HOST" "ls -la $REMOTE_PATH/pytorch/workspaces/emotion_regression/ 2>/dev/null"

echo ""
echo "📄 Checking for logs in emotion_regression..."
ssh "$REMOTE_USER@$REMOTE_HOST" "find $REMOTE_PATH/pytorch/workspaces/emotion_regression/ -name '*.log' -type f -exec ls -la {} \; 2>/dev/null"

echo ""
echo "💾 Checking for checkpoints in emotion_regression..."
ssh "$REMOTE_USER@$REMOTE_HOST" "find $REMOTE_PATH/pytorch/workspaces/emotion_regression/ -name '*.pth' -type f -exec ls -la {} \; 2>/dev/null"

echo ""
echo "📊 Checking for statistics in emotion_regression..."
ssh "$REMOTE_USER@$REMOTE_HOST" "find $REMOTE_PATH/pytorch/workspaces/emotion_regression/ -name '*.pkl' -type f -exec ls -la {} \; 2>/dev/null"

echo ""
echo "🔍 Checking what the long-running Python process is doing..."
ssh "$REMOTE_USER@$REMOTE_HOST" "ps -p 3081040 -o pid,ppid,cmd,etime,cputime 2>/dev/null"

echo ""
echo "📋 Checking process details and working directory..."
ssh "$REMOTE_USER@$REMOTE_HOST" "ls -la /proc/3081040/cwd 2>/dev/null"

echo ""
echo "📄 Checking if there are any recent log files anywhere..."
ssh "$REMOTE_USER@$REMOTE_HOST" "find $REMOTE_PATH -name '*.log' -type f -mtime -7 -exec ls -la {} \; 2>/dev/null"

echo ""
echo "🔄 Checking for any output files in the current directory..."
ssh "$REMOTE_USER@$REMOTE_HOST" "ls -la $REMOTE_PATH/*.out $REMOTE_PATH/*.log 2>/dev/null || echo 'No output files in main directory'"

echo ""
echo "📁 Checking all subdirectories for recent activity..."
ssh "$REMOTE_USER@$REMOTE_HOST" "find $REMOTE_PATH -type f -mtime -1 -name '*.log' -o -name '*.pth' -o -name '*.pkl' 2>/dev/null | head -20" 