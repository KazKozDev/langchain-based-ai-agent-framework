#!/bin/bash

echo "🍎 Installing ChromeDriver for Mac Studio M1"
echo "========================================="

# Check architecture
if [[ $(uname -m) != "arm64" ]]; then
    echo "⚠️  This script is intended for Apple Silicon Mac"
    exit 1
fi

echo "✅ Apple Silicon Mac detected"

# Check Homebrew
if ! command -v brew &> /dev/null; then
    echo "❌ Homebrew not found. Please install Homebrew first:"
    echo "   /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
    exit 1
fi

echo "✅ Homebrew found"

# Install Chrome if not installed
if [ ! -d "/Applications/Google Chrome.app" ]; then
    echo "📥 Installing Google Chrome..."
    brew install --cask google-chrome
else
    echo "✅ Google Chrome is already installed"
fi

# Install ChromeDriver
echo "📥 Installing ChromeDriver..."
brew install chromedriver

# Allow ChromeDriver execution (macOS security)
echo "🔓 Allowing ChromeDriver execution..."
chromedriver_path=$(which chromedriver)
if [ -n "$chromedriver_path" ]; then
    xattr -d com.apple.quarantine "$chromedriver_path" 2>/dev/null || true
    echo "✅ ChromeDriver allowed: $chromedriver_path"
else
    echo "⚠️  ChromeDriver not found in PATH"
fi

# Clear webdriver-manager cache
echo "🧹 Clearing webdriver-manager cache..."
rm -rf ~/.wdm

echo ""
echo "🎉 Installation complete!"
echo ""
echo "Now run the test:"
echo "python test_m1_mac.py"