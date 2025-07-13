#!/bin/bash

# This script saves all installed Python packages in the current environment
# to a requirements.txt file.

# The name of the output file.
REQUIREMENTS_FILE="requirements.txt"

# Use pip freeze to get the list of installed packages and their versions.
echo "Saving installed packages to '$REQUIREMENTS_FILE'..."
pip freeze > "$REQUIREMENTS_FILE"

# Check if the file was created successfully.
if [ $? -eq 0 ]; then
    echo "Successfully saved packages to '$REQUIREMENTS_FILE'. ✅"
else
    echo "An error occurred while saving the packages. ❌"
fi