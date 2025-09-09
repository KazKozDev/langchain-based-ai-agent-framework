#!/usr/bin/env python3
"""
Script to fix ChromeDriver issues on ARM Mac (Apple Silicon).
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def check_chrome_installation():
    """Check Chrome installation."""
    chrome_paths = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium"
    ]
    
    for path in chrome_paths:
        if os.path.exists(path):
            print(f"✅ Found Chrome: {path}")
            return True
    
    print("❌ Chrome not found. Please install Google Chrome.")
    return False

def check_system_info():
    """Show system information."""
    print(f"🖥️ System: {platform.system()}")
    print(f"🏗️ Architecture: {platform.machine()}")
    print(f"🐍 Python: {platform.python_version()}")
    
    # Check for ARM Mac
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        print("✅ ARM Mac (Apple Silicon) detected")
        return True
    return False

def install_chromedriver():
    """Install the correct ChromeDriver version for Mac Studio M1."""
    try:
        print("🔄 Updating webdriver-manager...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "webdriver-manager"])
        
        print("🔄 Installing ChromeDriver...")
        from webdriver_manager.chrome import ChromeDriverManager
        from webdriver_manager.core.utils import ChromeType
        
        # Clear cache
        cache_dir = Path.home() / ".wdm"
        if cache_dir.exists():
            import shutil
            shutil.rmtree(cache_dir)
            print("🗑️ Cache cleared")
        
        # Detect architecture
        is_apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"
        
        if is_apple_silicon:
            print("🍎 Apple Silicon Mac detected, using special installation")
            
            # Try different Chrome types
            chrome_types = [ChromeType.GOOGLE, ChromeType.CHROMIUM]
            driver_path = None
            
            for chrome_type in chrome_types:
                try:
                    print(f"📥 Trying install for {chrome_type.value}...")
                    driver_manager = ChromeDriverManager(chrome_type=chrome_type)
                    driver_path = driver_manager.install()
                    
                    # Ensure permissions
                    import stat
                    if os.path.exists(driver_path):
                        os.chmod(driver_path, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
                        print(f"✅ ChromeDriver installed for {chrome_type.value}: {driver_path}")
                        break
                        
                except Exception as e:
                    print(f"⚠️ Failed to install for {chrome_type.value}: {e}")
                    continue
            
            if not driver_path:
                raise Exception("Failed to install any ChromeDriver version")
                
        else:
            # Standard installation for non-Apple Silicon
            driver_manager = ChromeDriverManager()
            driver_path = driver_manager.install()
            
            # Ensure permissions
            import stat
            if os.path.exists(driver_path):
                current_permissions = os.stat(driver_path).st_mode
                os.chmod(driver_path, current_permissions | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
                print(f"✅ ChromeDriver installed: {driver_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ ChromeDriver installation error: {e}")
        return False

def test_chromedriver():
    """Test ChromeDriver."""
    try:
        print("🧪 Testing ChromeDriver...")
        
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service
        from webdriver_manager.chrome import ChromeDriverManager
        from webdriver_manager.core.utils import ChromeType
        
        chrome_options = Options()
        chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-web-security")
        chrome_options.add_argument("--disable-extensions")
        
        # For Apple Silicon try different Chrome types
        is_apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"
        
        driver = None
        if is_apple_silicon:
            print("🍎 Testing on Apple Silicon...")
            chrome_types = [ChromeType.GOOGLE, ChromeType.CHROMIUM]
            
            for chrome_type in chrome_types:
                try:
                    print(f"🔄 Trying {chrome_type.value}...")
                    service = Service(ChromeDriverManager(chrome_type=chrome_type).install())
                    driver = webdriver.Chrome(service=service, options=chrome_options)
                    break
                except Exception as e:
                    print(f"⚠️ {chrome_type.value} did not work: {e}")
                    continue
        else:
            # Standard test
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=chrome_options)
        
        if not driver:
            raise Exception("Failed to create WebDriver")
        
        # Simple test
        print("🌐 Loading Google...")
        driver.get("https://www.google.com")
        title = driver.title
        driver.quit()
        
        print(f"✅ Test successful! Page title: {title}")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        if 'driver' in locals() and driver:
            try:
                driver.quit()
            except:
                pass
        return False

def main():
    """Main function."""
    print("🔧 ChromeDriver diagnostics for ARM Mac\n")
    
    # Check system
    is_arm_mac = check_system_info()
    print()
    
    # Check Chrome
    if not check_chrome_installation():
        return
    print()
    
    # Install ChromeDriver
    if install_chromedriver():
        print()
        # Test
        if test_chromedriver():
            print("\n🎉 ChromeDriver is configured and working!")
        else:
            print("\n❌ ChromeDriver installed, but the test failed")
    else:
        print("\n❌ Failed to install ChromeDriver")

if __name__ == "__main__":
    main()