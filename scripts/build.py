#!/usr/bin/env python3
import os
import subprocess
import sys
from pathlib import Path

def run_npm_build():
    """Run npm build process"""
    try:
        result = subprocess.run(['npm', 'run', 'build'], 
                              capture_output=True, 
                              text=True, 
                              cwd=Path(__file__).parent.parent)
        
        if result.returncode != 0:
            print(f"Build failed: {result.stderr}")
            return False
        
        print("TypeScript compilation successful!")
        return True
        
    except FileNotFoundError:
        print("npm not found. Please install Node.js and npm.")
        return False
    except Exception as e:
        print(f"Build error: {e}")
        return False

if __name__ == "__main__":
    success = run_npm_build()
    sys.exit(0 if success else 1)