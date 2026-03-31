#!/usr/bin/env bash

# Recursively delete executable all files that have no extension (no dot in the filename)
# Hidden files (starting with .) are ignored for safety.

SCRIPT_PATH="$(realpath "$0")"

echo "Searching for executable files..."
FILES=$(find . -type f -perm /111 ! -samefile "$SCRIPT_PATH")

if [ -z "$FILES" ]; then
    echo "No executable files found."
    exit 0
fi

echo "The following executable files will be deleted:"
echo "$FILES"
echo

read -p "Proceed with deletion? (y/N): " confirm

if [[ "$confirm" == "y" || "$confirm" == "Y" ]]; then
    find . -type f -perm /111 ! -samefile "$SCRIPT_PATH" -delete
    echo "Deletion complete."
else
    echo "Aborted."
fi
