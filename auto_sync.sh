#!/bin/bash

# Kill any existing fswatch processes
pkill fswatch

# Start watching for changes
echo "Starting automatic sync..."
fswatch -o . | while read f; do
    echo "Change detected, syncing..."
    ./sync_to_remote.sh
    echo "Sync completed at $(date)"
done 