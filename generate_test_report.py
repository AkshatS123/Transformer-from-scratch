#!/usr/bin/env python3
"""
Generate a test report for the README
"""
import subprocess
import sys
import os

def run_tests_and_capture():
    """Run tests and capture output"""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(__file__)
        )
        
        if result.returncode == 0:
            print("✅ All tests passed!")
            print(f"Total tests: {result.stdout.count('PASSED')}")
            print("\nTest summary:")
            
            # Extract test names and results
            lines = result.stdout.split('\n')
            for line in lines:
                if 'PASSED' in line or 'FAILED' in line:
                    print(f"  {line}")
                    
            return True
        else:
            print("❌ Some tests failed")
            print(result.stdout)
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"Error running tests: {e}")
        return False

if __name__ == "__main__":
    success = run_tests_and_capture()
    sys.exit(0 if success else 1)