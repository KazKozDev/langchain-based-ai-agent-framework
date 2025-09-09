#!/usr/bin/env python3
"""
Script to clean and reinstall ChromeDriver on Mac Studio M1 (Apple Silicon).
"""

import os
import shutil
import platform
from pathlib import Path

def clean_chromedriver_cache():
    """Clean ChromeDriver cache."""
    print("🧹 Cleaning ChromeDriver cache...")
    
    # Main webdriver-manager cache locations
    cache_paths = [
        Path.home() / ".wdm",
        Path.home() / ".cache" / "selenium",
        "/tmp/.chromedriver-*",
    ]
    
    for cache_path in cache_paths:
        if isinstance(cache_path, Path) and cache_path.exists():
            try:
                shutil.rmtree(cache_path)
                print(f"✅ Removed cache: {cache_path}")
            except Exception as e:
                print(f"⚠️ Failed to remove {cache_path}: {e}")
    
    print("🧹 Cache cleaned!")

def main():
    print("🔧 ChromeDriver cleanup for Mac Studio M1")
    print(f"🖥️ System: {platform.system()} {platform.machine()}")
    
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        print("✅ Apple Silicon Mac detected")
        clean_chromedriver_cache()
        
        print("\n📋 Next steps:")
        print("1. Run: python fix_chromedriver.py")
        print("2. Then: python bitcoin_test.py")
        print("\nOr install ChromeDriver manually:")
        print("brew install --cask google-chrome")
        print("brew install chromedriver")
    else:
        print("ℹ️ This script is intended for Apple Silicon Mac")

if __name__ == "__main__":
    main()