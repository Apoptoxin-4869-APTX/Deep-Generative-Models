#!/bin/bash

# This script installs Python libraries from a requirements.txt file.

# The name of the requirements file.
# You can change this if your file has a different name.
REQUIREMENTS_FILE="requirements.txt"

# Check if the requirements file exists.
if [ ! -f "$REQUIREMENTS_FILE" ]; then
    echo "Error: '$REQUIREMENTS_FILE' not found."
    exit 1
fi

# Install the libraries using pip.
echo "Installing libraries from '$REQUIREMENTS_FILE'..."
pip install -r "$REQUIREMENTS_FILE"

# Check if the installation was successful.
if [ $? -eq 0 ]; then
    echo "All libraries were installed successfully. ✅"
else
    echo "An error occurred during installation. ❌"
fi