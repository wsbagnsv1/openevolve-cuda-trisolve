import os
import sys
import subprocess
import re

def main():
    # Configuration
    base_dir = os.path.dirname(os.path.abspath(__file__))
    examples_dir = os.path.join(base_dir, "examples", "cuda_solve_tri_opt")
    checkpoints_dir = os.path.join(examples_dir, "openevolve_output", "checkpoints")
    
    initial_program = os.path.join(examples_dir, "initial_program.py")
    evaluator = os.path.join(examples_dir, "evaluator.py")
    config = os.path.join(examples_dir, "config.yaml")
    runner_script = os.path.join(base_dir, "openevolve-run.py")

    # Find latest checkpoint
    latest_checkpoint = None
    if os.path.exists(checkpoints_dir):
        checkpoints = []
        for d in os.listdir(checkpoints_dir):
            if d.startswith("checkpoint_") and os.path.isdir(os.path.join(checkpoints_dir, d)):
                try:
                    iteration = int(d.split("_")[1])
                    checkpoints.append((iteration, os.path.join(checkpoints_dir, d)))
                except ValueError:
                    continue
        
        if checkpoints:
            # Sort by iteration number (descending)
            checkpoints.sort(key=lambda x: x[0], reverse=True)
            latest_checkpoint = checkpoints[0][1]
            print(f"Found latest checkpoint: {os.path.basename(latest_checkpoint)} (Iteration {checkpoints[0][0]})")

    # Construct command
    cmd = [sys.executable, runner_script, initial_program, evaluator, "--config", config]
    
    if latest_checkpoint:
        cmd.extend(["--checkpoint", latest_checkpoint])
    else:
        print("No checkpoints found. Starting from scratch.")

    # Run command
    print(f"Executing: {' '.join(cmd)}")
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        print(f"Evolution process exited with error code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\nEvolution stopped by user.")
        sys.exit(0)

if __name__ == "__main__":
    main()
