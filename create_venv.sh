#!/bin/bash

# Script to create a new Python virtual environment named 'dgm'

echo "--- Virtual Environment Creation Script ---"

# 1. Check if python3 is installed
if ! command -v python3 &> /dev/null
then
    echo "Error: python3 is not installed. Please install it first."
    echo "On Debian/Ubuntu: sudo apt update && sudo apt install python3 python3-venv"
    echo "On Fedora/RHEL: sudo dnf install python3 python3-venv"
    echo "On Arch Linux: sudo pacman -S python python-virtualenv"
    exit 1
fi

# 2. Check if venv module is available for python3
#    This is typically installed with python3-venv package on Debian/Ubuntu, or part of python3 on other distros.
if ! python3 -c "import venv" &> /dev/null
then
    echo "Error: Python 'venv' module not found. Please ensure it's installed."
    echo "On Debian/Ubuntu, you might need to install 'python3-venv': sudo apt install python3-venv"
    echo "On other systems, it might be part of the main python3 installation, or requires a separate package."
    exit 1
fi

# 3. Define the virtual environment name
VENV_NAME="dgm"

# 4. Check if the directory already exists
if [ -d "$VENV_NAME" ]; then
    echo "Warning: A directory named '$VENV_NAME' already exists in the current location."
    read -p "Do you want to remove it and create a new one? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]
    then
        echo "Removing existing '$VENV_NAME' directory..."
        rm -rf "$VENV_NAME"
    else
        echo "Aborting. Virtual environment not created."
        exit 0
    fi
fi

# 5. Create the virtual environment
echo "Creating virtual environment '$VENV_NAME'..."
python3 -m venv "$VENV_NAME"

# Check if the creation was successful
if [ $? -eq 0 ]; then
    echo "Virtual environment '$VENV_NAME' created successfully!"
    echo "To activate it, run: source $VENV_NAME/bin/activate"
    echo "To deactivate it, run: deactivate"
else
    echo "Error: Failed to create virtual environment '$VENV_NAME'."
    exit 1
fi

echo "--- Script Finished ---"