import subprocess
import os

def run_script(script_path):
    print(f"\n--- Running {script_path} ---")
    result = subprocess.run(["python", script_path], capture_output=False, text=True)
    if result.returncode == 0:
        print(f"Successfully finished {script_path}")
    else:
        print(f"Error in {script_path}")

def main():
    # 1. Convert models to ONNX
    run_script("scripts/convert_to_onnx.py")
    
    # 2. Evaluate and update analysis.md
    run_script("scripts/evaluate_models.py")

if __name__ == "__main__":
    main()
