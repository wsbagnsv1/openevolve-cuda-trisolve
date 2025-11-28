import os
import subprocess
import tempfile
import re
import importlib.util
import yaml
import math
import shutil
import datetime
from openevolve.evaluation_result import EvaluationResult


# Metrics configuration from filter_report.py
METRICS_CONFIG = {
    # --- 1. THE COST (Overhead) ---
    "Executed Instructions": {
        "str": '"Instruction Statistics","Executed Instructions","inst","',
        "weight": 2.5,   # INCREASED: Penalize instruction bloat significantly
        "higher_is_better": False,
        "use_for_score": True
    },
    
    # --- 2. THE POTENTIAL BENEFIT (Memory Efficiency) ---
    "L2 Hit Rate": {
        "str": '"Memory Workload Analysis","L2 Hit Rate","%","',
        "weight": 3.0,   # KEEP HIGH: This is your most honest structural metric
        "higher_is_better": True,
        "use_for_score": True
    },
    "L1/TEX Cache Throughput": {
        "str": '"GPU Speed Of Light Throughput","L1/TEX Cache Throughput","%","',
        "weight": 1.0,
        "higher_is_better": True,
        "use_for_score": True
    },
    
    # --- 3. THE SAFETY RAIL (Speed & Utilization) ---
    "Duration": {
        "str": '"GPU Speed Of Light Throughput","Duration","ns","',
        "weight": 4.0,   # INCREASED: The ultimate veto. If this doubles, score tanks.
        "higher_is_better": False,
        "use_for_score": True
    },
    "Compute (SM) Throughput": {
        "str": '"GPU Speed Of Light Throughput","Compute (SM) Throughput","%","',
        "weight": 1.5,
        "higher_is_better": True,
        "use_for_score": True
    },

    # --- 4. THE STABLE BASELINE (Always Optimize) ---
    "Eligible Warps Per Scheduler": {
        "str": '"Scheduler Statistics","Eligible Warps Per Scheduler","warp","',
        "weight": 1.5,   # SLIGHT INCREASE: Reward breaking the 0.140 barrier
        "higher_is_better": True,
        "use_for_score": True
    },
    "Avg. Active Threads": {
        "str": '"Warp State Statistics","Avg. Active Threads Per Warp","","',
        "weight": 1.0,   # LOWERED: Usually static at 32, rarely differentiates
        "higher_is_better": True,
        "use_for_score": True
    },
    "Avg. Divergent Branches": {
        "str": '"Source Counters","Avg. Divergent Branches","","',
        "weight": 1.0,
        "higher_is_better": False,
        "use_for_score": True
    },
    
    # --- 5. MINOR INDICATORS ---
    "Shared Memory Throughput": {
        "str": '"Memory Workload Analysis","Shared Memory Throughput","byte/s","', 
        "weight": 0.5,
        "higher_is_better": True,
        "use_for_score": True
    },
    "DRAM Throughput": {
        "str": '"Memory Workload Analysis","Memory Throughput","byte/s","',
        "weight": 0.5, 
        "higher_is_better": False, # NOTE: Usually high throughput is good, but keeping your pref.
        "use_for_score": True
    },

    # --- 6. STRUCTURAL LIMITS ---
    "Registers Per Thread": {
        "str": '"Launch Statistics","Registers Per Thread","register/thread","',
        "weight": 1.0, 
        "higher_is_better": False,
        "use_for_score": True
    },
    "Warp Cycles Per Inst": {
        "str": '"Warp State Statistics","Warp Cycles Per Issued Instruction","cycle","',
        "weight": 3.0,   # LOWERED: From 5.0 to 3.0 to reduce "busy work" bias
        "higher_is_better": False,
        "use_for_score": True
    },
    "Theoretical Occupancy": {
        "str": '"Occupancy","Theoretical Occupancy","%","',
        "weight": 0.5,   # LOWERED: It's a ceiling, not a performance guarantee
        "higher_is_better": True,
        "use_for_score": True
    }
}

def calculate_combined_score(csv_data):
    """
    Calculate balanced score from Nsight Compute CSV data.
    Returns: (combined_score, metrics_dict)
    """
    results = {name: [] for name in METRICS_CONFIG}
    
    # Parse CSV data
    for line in csv_data.split('\n'):
        line = line.strip()
        for name, config in METRICS_CONFIG.items():
            search_string = config["str"]
            if search_string in line:
                try:
                    parts = line.split(search_string)
                    if len(parts) > 1:
                        value_str = parts[1].split('",')[0].replace('.', '').replace(',', '.')
                        val = float(value_str)
                        results[name].append(val)
                except ValueError:
                    pass
    
    # Calculate averages
    averages = {}
    for name in sorted(results.keys()):
        values = results[name]
        if values:
            avg = sum(values) / len(values)
            averages[name] = avg
    
    # Calculate balanced score using weighted logarithmic formula
    score = 0
    for name, config in METRICS_CONFIG.items():
        if not config["use_for_score"]:
            continue
            
        val = averages.get(name)
        if val is None:
            val = 0.000001
        safe_val = val if val > 0.000001 else 0.000001
        
        term = config["weight"] * math.log10(safe_val)
        
        if config["higher_is_better"]:
            score += term
        else:
            score -= term
    
    # Offset for display
    combined_score = score + 50
    
    return combined_score, averages


def save_program_to_logs(program_path):
    """Save the program being evaluated to logs with timestamp-based filename."""
    try:
        # Find the logs directory
        example_dir = os.path.dirname(os.path.abspath(__file__))
        logs_dir = os.path.join(example_dir, "openevolve_output", "logs")
        
        # Create logs directory if it doesn't exist
        os.makedirs(logs_dir, exist_ok=True)
        
        # Generate timestamp-based filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        destination = os.path.join(logs_dir, f"program_{timestamp}.py")
        
        # Copy the program
        shutil.copy2(program_path, destination)
        print(f"Saved program as: program_{timestamp}.py")
    except Exception as e:
        print(f"Warning: Could not save program to logs: {e}")


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
    # Save program to logs directory with timestamp
    save_program_to_logs(program_path)
    
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

    # 4. Performance Benchmark (Console-Only Information)
    # Single run on dev1 for informational purposes - not sent to LLM
    perf_cmd = [test_backend_ops, "perf", "-o", "SOLVE_TRI"]
    
    print(f"--- Performance Benchmark (Dev1 Only - Console Info) ---")
    
    try:
        # Set environment to only use dev1
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "1"
        
        perf_result = subprocess.run(
            perf_cmd,
            cwd=os.path.dirname(test_backend_ops),
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
            env=env
        )
        
        if perf_result.returncode != 0:
            print(f"Warning: Performance benchmark failed (exit code {perf_result.returncode})")
            if perf_result.stderr:
                print(perf_result.stderr)
        else:
            # Log full output for console info
            print(perf_result.stdout)

    except subprocess.TimeoutExpired:
        print("Warning: Performance benchmark timed out")
    except Exception as e:
        print(f"Warning: Performance benchmark failed: {e}")

    # 5. Nsight Compute Profiling (Dev1 Only)
    nsight_profile = ""
    try:
        print("--- Starting Nsight Compute Profiling (Dev1 Only) ---")
        
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
        
        # Profile only Device 1 (RTX 2070)
        device_id = 1
        # Use absolute path for report
        report_path = os.path.join(os.path.dirname(test_backend_ops), f"nsight_profile_dev{device_id}.csv")
        
        # Run Nsight Compute profiling with CUDA_VISIBLE_DEVICES
        ncu_cmd = [
            ncu_exe,
            "--csv",
            "--log-file", report_path,
            "--set", "full",
            "--launch-count", "50",
            test_backend_ops,
            "perf",
            "-o", "SOLVE_TRI"
        ]
        
        # Set environment to profile only dev1
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
            print(f"Warning: Profiling completed with code {ncu_result.returncode}")
        
        # Read the CSV
        if os.path.exists(report_path):
            with open(report_path, 'r', encoding='utf-8') as f:
                nsight_profile = f.read()
            
            lines = nsight_profile.strip().split('\n')
            data_rows = len(lines) - 1 if len(lines) > 1 else 0
            print(f"--- Nsight Profiling Complete ---")
            print(f"Collected {data_rows} data rows")
            print(f"Report size: {len(nsight_profile)} bytes")
            
            # Clean up temp file
            try:
                os.remove(report_path)
            except:
                pass
        else:
            print(f"Warning: Profile report not found")
            nsight_profile = "Profile report not generated"
            
    except subprocess.TimeoutExpired:
        print("Warning: Nsight profiling timed out")
        nsight_profile = "Profiling timed out after 180 seconds"
    except FileNotFoundError:
        print("Warning: 'ncu' command not found. Skipping Nsight profiling.")
        nsight_profile = "Nsight Compute (ncu) not found in PATH"
    except Exception as e:
        print(f"Warning: Nsight profiling failed: {e}")
        nsight_profile = f"Profiling failed: {str(e)}"

    # Calculate balanced score from Nsight profile
    if nsight_profile and nsight_profile not in ["Profile report not generated", "Profiling timed out after 180 seconds", "Nsight Compute (ncu) not found in PATH"]:
        combined_score, metric_averages = calculate_combined_score(nsight_profile)
        
        # Print all metrics for console
        print("\n=== METRICS ===")
        for name in sorted(metric_averages.keys()):
            avg = metric_averages[name]
            if avg > 1000:
                print(f"{name}: {avg:,.3f}")
            else:
                print(f"{name}: {avg:.3f}")
        
        print(f"\n=== BALANCED SCORE ===")
        print(f"Score: {combined_score:.4f}")
        print("(Weighted score: rewards speed, memory efficiency, and occupancy)")
        
        # Build metrics dict for LLM (combined_score + individual metrics)
        metrics = {"combined_score": combined_score}
        for name, avg in metric_averages.items():
            # Use snake_case for metric keys
            metric_key = name.lower().replace(" ", "_").replace(".", "").replace("/", "_")
            metrics[metric_key] = avg
        
        return EvaluationResult(
            metrics=metrics,
            artifacts={"stdout": f"Balanced score calculated from {len(metric_averages)} Nsight metrics"}
        )
    else:
        # Fallback if profiling failed
        print("\n=== WARNING ===")
        print("Nsight profiling failed - cannot calculate balanced score")
        return EvaluationResult(
            metrics={"error": "Nsight profiling failed", "combined_score": 0.0},
            artifacts={"stderr": nsight_profile}
        )



if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python evaluator.py <program_path>")
        sys.exit(1)
    
    result = evaluate(sys.argv[1])
    print(result)
