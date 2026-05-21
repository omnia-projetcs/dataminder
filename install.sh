#!/bin/bash

echo "=========================================="
echo "Dataminder Dependencies Installation"
echo "=========================================="

OS="$(uname -s)"
echo -e "\n[1/3] Installing system packages..."
if [ "$OS" = "Linux" ]; then
    echo "Linux detected. Using apt-get (requires administrator rights)..."
    sudo apt-get update
    sudo apt-get install -y tesseract-ocr tesseract-ocr-eng tesseract-ocr-fra poppler-utils antiword libchm-bin ffmpeg
elif [ "$OS" = "Darwin" ]; then
    echo "macOS detected. Using Homebrew..."
    if ! command -v brew &> /dev/null; then
        echo "Error: Homebrew is not installed. Please install Homebrew first (https://brew.sh/)."
        exit 1
    fi
    brew install tesseract tesseract-lang poppler antiword chmlib ffmpeg
else
    echo "Unsupported OS: $OS. Please install dependencies manually."
    exit 1
fi

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

echo -e "\n[3/3] Downloading the Ollama model (gemma3:12b)..."
echo "Please make sure Ollama is running in the background or another terminal."
ollama pull gemma3:12b || echo "Failed to pull the model. You can run 'ollama pull gemma3:12b' manually later."

echo -e "\n=========================================="
echo "Installation completed successfully!"
echo "Before running the application, don't forget to activate the virtual environment if not already done:"
echo "source venv/bin/activate"
echo "Then run:"
echo "python main.py"
echo "=========================================="
