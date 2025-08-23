#!/bin/bash
# Review before running; destructive.
#
# macOS Artifact Cleanup Script
# Removes __MACOSX directories created by macOS archive extraction
# 
# Usage: bash scripts/cleanup_remove_macos_artifacts.sh
# 
# This script will permanently delete macOS metadata artifacts.
# Make sure you have backups if needed before running.

set -euo pipefail

echo "=== macOS Artifact Cleanup Script ==="
echo "Date: $(date)"
echo

# Check for ATOFSIMSCLASS/__MACOSX directory
MACOS_PATH="ATOFSIMSCLASS/__MACOSX"

if [ -d "$MACOS_PATH" ]; then
    echo "Found macOS artifacts at: $MACOS_PATH"
    echo "Calculating size..."
    SIZE=$(du -sh "$MACOS_PATH" 2>/dev/null | cut -f1 || echo "unknown")
    echo "Size: $SIZE"
    echo
    echo "Removing $MACOS_PATH..."
    rm -rf "$MACOS_PATH"
    echo "✅ Successfully removed: $MACOS_PATH"
else
    echo "ℹ️  No macOS artifacts found at: $MACOS_PATH"
fi

echo
echo "=== Cleanup Complete ==="
echo "Actions taken logged to console output"