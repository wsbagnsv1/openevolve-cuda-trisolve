import os
import subprocess
import tempfile
import re
import importlib.util
import yaml
from openevolve.evaluation_result import EvaluationResult


def load_config():
    """Load configuration from config.yaml"""
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Warning: Could not load config.yaml: {e}")
        return {}


def evaluate(program_path):
    """
    Evaluates the evolved CUDA kernel by integrating it into llama.cpp and running benchmarks.
    Args:
        program_path (str): Path to the python file containing the evolved kernel.
    """
    # Sanitize the file content before importing
    try:
        with open(program_path, "r", encoding='utf-8') as f:
            content = f.read()
        
        lines = content.splitlines()
        original_len = len(lines)
        
        # Aggressively strip leading artifacts (empty lines, 'python', markdown fences)
        while lines:
            first_line = lines[0].strip()
            if not first_line:  # Empty line
                lines.pop(0)
                continue
            if first_line.lower() == "python":
                lines.pop(0)
                continue
            if first_line.startswith("```"):
                lines.pop(0)
                continue
            # If we reach here, it's a valid line (docstring, import, etc.)
            break
            
        # Filter out EVOLVE-BLOCK markers
        lines = [line for line in lines if "# EVOLVE-BLOCK-START" not in line and "# EVOLVE-BLOCK-END" not in line]
            
        # Also strip trailing lines
        while lines:
            last_line = lines[-1].strip()
            if not last_line:
                lines.pop()
                continue
            if last_line.startswith("```"):
                lines.pop()
                continue
            break
            
        if len(lines) < original_len:
            new_content = "\n".join(lines)
            with open(program_path, "w", encoding='utf-8') as f:
                f.write(new_content)
                
    except Exception as e:
        print(f"Warning: Failed to sanitize file: {e}")

    # Load the program module
    try:
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)
    except Exception as e:
        error_msg = f"Module import failed: {str(e)}"
        print(error_msg)
        
        # Save failed program for debugging
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_dir = os.path.join(os.path.dirname(program_path), "failed_programs")
        os.makedirs(debug_dir, exist_ok=True)
        
        debug_path = os.path.join(debug_dir, f"failed_import_{timestamp}.py")
        try:
            import shutil
            shutil.copy2(program_path, debug_path)
            print(f"Failed program saved to: {debug_path}")
        except Exception as save_error:
            print(f"Warning: Could not save failed program: {save_error}")
        
        return EvaluationResult(
            metrics={"error": "Module import failed", "combined_score": 0.0},
            artifacts={"stderr": error_msg}
        )

    if not hasattr(program, "cuda_source"):
        return EvaluationResult(
            metrics={"error": "cuda_source variable not found", "combined_score": 0.0},
            artifacts={"stderr": "The program must define a 'cuda_source' string variable containing the CUDA kernel code."}
        )

    cuda_source = program.cuda_source
    
    # Load configuration
    config = load_config()
    paths_config = config.get('paths', {})
    
    # Get paths from config with fallback to defaults
    llama_cpp_root = paths_config.get('llama_cpp_root', r"F:\Users\timbe\Desktop\test optimization")
    target_kernel_file = paths_config.get('target_kernel_file', r"ggml\src\ggml-cuda\solve_tri.cu")
    build_dir_name = paths_config.get('build_dir', "build")
    test_exe_path = paths_config.get('test_executable', r"bin\Release\test-backend-ops.exe")
    
    # Construct full paths
    target_file = os.path.join(llama_cpp_root, target_kernel_file)
    build_dir = os.path.join(llama_cpp_root, build_dir_name)
    test_backend_ops = os.path.join(build_dir, test_exe_path)

    # 1. Write the kernel to the target file
    try:
        print(f"--- Writing Kernel ---")
        print(f"Target file: {target_file}")
        with open(target_file, "w", encoding='utf-8') as f:
            f.write(cuda_source)
        print(f"Successfully wrote {len(cuda_source)} bytes to {target_file}")
    except Exception as e:
        return EvaluationResult(
            metrics={"error": "Failed to write kernel file", "combined_score": 0.0},
            artifacts={"stderr": str(e)}
        )

    # 2. Compile
    # Configure with -lineinfo for Nsight debugging
    config_cmd = ["cmake", "-B", "build", "-DCMAKE_CUDA_FLAGS=-lineinfo"]
    compile_cmd = ["cmake", "--build", "build", "--config", "Release", "-j"]
    
    try:
        # Run configuration first
        config_result = subprocess.run(
            config_cmd,
            cwd=llama_cpp_root,
            capture_output=True,
            text=True,
            check=False
        )
        
        if config_result.returncode != 0:
             return EvaluationResult(
                metrics={"error": "CMake configuration failed", "combined_score": 0.0},
                artifacts={"stderr": config_result.stderr, "stdout": config_result.stdout}
            )
        
        print("--- Configuration Complete ---")

        compile_result = subprocess.run(
            compile_cmd,
            cwd=llama_cpp_root,
            capture_output=True,
            text=True,
            check=False
        )
        
        if compile_result.returncode != 0:
            return EvaluationResult(
                metrics={"error": "Compilation failed", "combined_score": 0.0},
                artifacts={"stderr": compile_result.stderr, "stdout": compile_result.stdout}
            )
        
        print("--- Compilation Complete ---")

    except Exception as e:
        return EvaluationResult(
            metrics={"error": "Compilation execution failed", "combined_score": 0.0},
            artifacts={"stderr": str(e)}
        )

    # 3. Correctness Check
    correctness_cmd = [test_backend_ops, "-o", "SOLVE_TRI"]
    try:
        correctness_result = subprocess.run(
            correctness_cmd,
            # Run from the directory where the exe is to ensure it finds necessary DLLs/resources
            cwd=os.path.dirname(test_backend_ops),
            capture_output=True,
            text=True,
            timeout=120,
            check=False
        )

        # print("--- Correctness Check Output ---")
        # print(correctness_result.stdout)
        # if correctness_result.stderr:
        #     print("--- Correctness Check Errors ---")
        #     print(correctness_result.stderr)

        if correctness_result.returncode != 0:
             return EvaluationResult(
                metrics={"error": "Correctness check failed (nonzero exit)", "combined_score": 0.0},
                artifacts={"stderr": correctness_result.stderr, "stdout": correctness_result.stdout}
            )
        
        # Check for "tests passed" or "OK" in output
        # User requirement: Parse "X/Y tests passed" and "X/Y backends passed"
        # Ensure X == Y for all cases.
        
        tests_passed_matches = re.findall(r"(\d+)/(\d+) tests passed", correctness_result.stdout)
        backends_passed_matches = re.findall(r"(\d+)/(\d+) backends passed", correctness_result.stdout)
        
        # Removed debug logging as requested

        if not tests_passed_matches and not backends_passed_matches:
             # Fallback if output format is completely unexpected
             if "OK" not in correctness_result.stdout:
                 return EvaluationResult(
                    metrics={"error": "Correctness check failed (no 'tests passed' or 'backends passed' found)", "combined_score": 0.0},
                    artifacts={"stdout": correctness_result.stdout}
                )
        
        # Verify tests passed
        for passed, total in tests_passed_matches:
            if passed != total:
                 return EvaluationResult(
                    metrics={"error": f"Correctness check failed: {passed}/{total} tests passed", "combined_score": 0.0},
                    artifacts={"stdout": correctness_result.stdout}
                )
        
        # Verify backends passed
        for passed, total in backends_passed_matches:
            if passed != total:
                 return EvaluationResult(
                    metrics={"error": f"Correctness check failed: {passed}/{total} backends passed", "combined_score": 0.0},
                    artifacts={"stdout": correctness_result.stdout}
                )
        
        print("--- Correctness Check Passed ---")

    except subprocess.TimeoutExpired:
        return EvaluationResult(
            metrics={"error": "Correctness check timeout", "combined_score": 0.0},
            artifacts={"stderr": "Correctness check took too long."}
        )
    except Exception as e:
        return EvaluationResult(
            metrics={"error": "Correctness check execution failed", "combined_score": 0.0},
            artifacts={"stderr": str(e)}
        )

    # 4. Performance Benchmark
    perf_cmd = [test_backend_ops, "perf", "-o", "SOLVE_TRI"]
    
    num_runs = 25
    total_throughput_score = 0.0
    
    # List of lists to store us/run for each device across runs
    # We don't know how many devices yet, will initialize on first run
    per_device_us_runs = [] 
    
    print(f"--- Starting {num_runs}x Performance Benchmark ---")
    
    for run_idx in range(num_runs):
        try:
            perf_result = subprocess.run(
                perf_cmd,
                cwd=os.path.dirname(test_backend_ops),
                capture_output=True,
                text=True,
                timeout=256,
                check=False
            )
            
            if perf_result.returncode != 0:
                print(f"Run {run_idx+1}/{num_runs} failed.")
                if perf_result.stderr:
                    print(perf_result.stderr)
                return EvaluationResult(
                    metrics={"error": f"Performance benchmark failed on run {run_idx+1}", "combined_score": 0.0},
                    artifacts={"stderr": perf_result.stderr, "stdout": perf_result.stdout}
                )
            
            
            # Log first run output for debugging
            # if run_idx == 0:
            #     print("--- Performance Benchmark Output (Run 1) ---")
            #     print(perf_result.stdout)
            #     if perf_result.stderr:
            #         print("--- Performance Benchmark Errors/Warnings (Run 1) ---")
            #         print(perf_result.stderr)

            # Parse output
            us_runs = re.findall(r"([\d\.]+)\s+us/run", perf_result.stdout)
            
            if not us_runs:
                 print(f"Run {run_idx+1}/{num_runs}: No performance data found.")
                 return EvaluationResult(
                    metrics={"error": "No performance data found", "combined_score": 0.0},
                    artifacts={"stdout": perf_result.stdout}
                )
            
            # Convert to floats
            us_runs = [float(x) for x in us_runs]
            
            # Initialize per_device storage if needed
            if not per_device_us_runs:
                per_device_us_runs = [[] for _ in us_runs]
            
            # Store values
            for dev_idx, val in enumerate(us_runs):
                if dev_idx < len(per_device_us_runs):
                    per_device_us_runs[dev_idx].append(val)
            
            # Removed per-run logging as requested

        except subprocess.TimeoutExpired:
            return EvaluationResult(
                metrics={"error": "Performance benchmark timeout", "combined_score": 0.0},
                artifacts={"stderr": "Performance benchmark took too long."}
            )
        except Exception as e:
            return EvaluationResult(
                metrics={"error": "Performance benchmark execution failed", "combined_score": 0.0},
                artifacts={"stderr": str(e)}
            )

    # Calculate averages and scores
    avg_us_per_device = []
    device_scores = []
    metrics = {}
    
    for dev_idx, runs in enumerate(per_device_us_runs):
        if runs:
            avg_us = sum(runs) / len(runs)
            avg_us_per_device.append(avg_us)
            # Include metrics and scores for all devices
            metrics[f"us_per_run_dev{dev_idx}"] = avg_us
            dev_score = 1000.0 / (avg_us + 1e-6)
            device_scores.append(dev_score)
        else:
            avg_us_per_device.append(0.0)

    # Combined score is the average of per-device scores (50:50 weight if 2 devices)
    if device_scores:
        combined_score = sum(device_scores) / len(device_scores)
    else:
        combined_score = 0.0

    print(f"--- Benchmark Complete ---")
    print(f"Per-device averages (us/run): {avg_us_per_device}")
    print(f"Per-device scores: {device_scores}")
    print(f"Combined Score: {combined_score:.4f}")

    # 5. Nsight Compute Profiling
    nsight_profile = ""
    try:
        print("--- Starting Nsight Compute Profiling ---")
        
        # Find ncu executable
        ncu_exe = "ncu"
        
        # Try to find ncu in PATH first
        try:
            subprocess.run(["ncu", "--version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            # If not in PATH, try config-specified location
            default_ncu_path = paths_config.get('nsight_compute_path', r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.1.0\ncu.bat")
            if os.path.exists(default_ncu_path):
                ncu_exe = default_ncu_path
            else:
                # Try to find any Nsight Compute installation in base directory
                nsight_base = paths_config.get('nsight_base_dir', r"C:\Program Files\NVIDIA Corporation")
                if os.path.exists(nsight_base):
                    for item in os.listdir(nsight_base):
                        if item.startswith("Nsight Compute"):
                            potential_ncu = os.path.join(nsight_base, item, "ncu.bat")
                            if os.path.exists(potential_ncu):
                                ncu_exe = potential_ncu
                                break
        
        # Profile both CUDA devices separately
        combined_profile = ""
        csv_header = None
        all_data_rows = []
        
        for device_id in [0, 1]:  # CUDA:0 (RTX 4070 Ti) and CUDA:1 (RTX 2070)
            device_name = "RTX 4070 Ti" if device_id == 0 else "RTX 2070"
            print(f"--- Profiling Device {device_id} ({device_name}) ---")
            
            # Create temp path for this device's report
            report_path = os.path.join(tempfile.gettempdir(), f"nsight_profile_dev{device_id}.csv")
            
            # Run Nsight Compute profiling with CUDA_VISIBLE_DEVICES
            ncu_cmd = [
                ncu_exe,
                "--csv",
                "--log-file", report_path,
                "--set", "full",
                "--launch-count", "1",
                test_backend_ops,
                "perf",
                "-o", "SOLVE_TRI"
            ]
            
            # Set environment to profile only this device
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(device_id)
            
            ncu_result = subprocess.run(
                ncu_cmd,
                cwd=os.path.dirname(test_backend_ops),
                capture_output=True,
                text=True,
                timeout=180,  # 3 minutes timeout for profiling
                check=False,
                env=env
            )
            
            if ncu_result.returncode != 0:
                print(f"Warning: Device {device_id} profiling completed with code {ncu_result.returncode}")
            
            # Read and parse this device's CSV
            if os.path.exists(report_path):
                with open(report_path, 'r', encoding='utf-8') as f:
                    device_csv = f.read()
                
                # Split into lines and separate header from data
                lines = device_csv.strip().split('\n')
                if lines:
                    if csv_header is None:
                        # First device - save the header
                        csv_header = lines[0]
                    
                    # Collect data rows (skip header line)
                    if len(lines) > 1:
                        all_data_rows.extend(lines[1:])
                
                print(f"Device {device_id}: {len(lines)-1 if len(lines) > 1 else 0} data rows")
                
                # Clean up temp file
                try:
                    os.remove(report_path)
                except:
                    pass
            else:
                print(f"Warning: Device {device_id} profile report not found")
        
        # Combine all CSV data
        if csv_header and all_data_rows:
            combined_profile = csv_header + '\n' + '\n'.join(all_data_rows)
            print(f"--- Nsight Profiling Complete ---")
            print(f"Combined profile: {len(all_data_rows)} total data rows from 2 devices")
            print(f"Total report size: {len(combined_profile)} bytes")
        else:
            print("Warning: No Nsight profile data collected")
            combined_profile = "Profile report not generated"
        
        nsight_profile = combined_profile
            
    except subprocess.TimeoutExpired:
        print("Warning: Nsight profiling timed out")
        nsight_profile = "Profiling timed out after 180 seconds"
    except FileNotFoundError:
        print("Warning: 'ncu' command not found. Skipping Nsight profiling.")
        nsight_profile = "Nsight Compute (ncu) not found in PATH"
    except Exception as e:
        print(f"Warning: Nsight profiling failed: {e}")
        nsight_profile = f"Profiling failed: {str(e)}"

    return EvaluationResult(
        metrics={**metrics, "combined_score": combined_score},
        artifacts={
            "stdout": f"Averaged over {num_runs} runs. Per-device averages: {avg_us_per_device}",
            "nsight_profile": nsight_profile
        }
    )


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python evaluator.py <program_path>")
        sys.exit(1)
    
    result = evaluate(sys.argv[1])
    print(result)
