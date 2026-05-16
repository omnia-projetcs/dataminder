#!/bin/bash

echo "=========================================="
echo "Dataminder Dependencies Installation"
echo "=========================================="

echo -e "\n[1/2] Installing system packages (requires administrator rights)..."
# Requests sudo rights if necessary
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-eng tesseract-ocr-fra poppler-utils antiword libchm-bin

echo -e "\n[2/3] Installing Python packages (Virtual Environment)..."
# Checks if the venv folder exists, creates it otherwise
if [ ! -d "venv" ]; then
    echo "Creating 'venv' virtual environment..."
    python3 -m venv venv
fi

echo "Activating the virtual environment..."
source venv/bin/activate

# Upgrades pip to avoid warnings
echo "Upgrading pip..."
pip install --upgrade pip

# Installs the Python dependencies listed in requirements.txt
pip install -r requirements.txt

echo -e "\n[3/3] Downloading the Ollama model (ministral-3:8b)..."
echo "Please make sure Ollama is running in the background or another terminal."
ollama pull ministral-3:8b || echo "Failed to pull the model. You can run 'ollama pull ministral-3:8b' manually later."

echo -e "\n=========================================="
echo "Installation completed successfully!"
echo "Before running the application, don't forget to activate the virtual environment if not already done:"
echo "source venv/bin/activate"
echo "Then run:"
echo "python main.py"
echo "=========================================="
