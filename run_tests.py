#!/usr/bin/env python3
"""
trying out simple test runner for transformer project
"""


#THIS IS SO ANNOYING TO DEBUG
import subprocess
import sys
import os


def run_tests():
    """Run all tests in the project"""
    print("🧪 Running transformer tests...")
    
    # pytest on tests directory
    try:
        result = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ All tests passed!")
        else:
            print("❌ Some tests failed:")
            print(result.stdout)
            print(result.stderr)
            
    except FileNotFoundError:
        print("❌ pytest not found. Install with: pip install pytest")


if __name__ == "__main__":
    run_tests() 