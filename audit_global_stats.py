# audit_global_stats.py (V4.1 Precision Edition)
import os
import re

def run_global_stat_scan():
    target_dir = 'src/features'
    # List of statistical functions that are dangerous when called on a full column
    dangerous_functions = ['mean', 'std', 'max', 'min', 'rank', 'quantile', 'median', 'sum']
    
    # List of safety prefixes that make the call "Causal" (Looking at history)
    safety_prefixes = [r'\.rolling\(', r'\.ewm\(', r'\.expanding\(', r'\.groupby\(']
    
    # List of safety markers that make the call "Horizontal" (Looking only at the current row)
    row_wise_markers = ['axis=1', "axis='columns'", 'axis="columns"']
    
    print(f"🕵️  STARTING REFINED GLOBAL STATISTIC SCAN: {target_dir}")
    print("-" * 60)
    
    violation_count = 0
    files_scanned = 0

    for root, _, files in os.walk(target_dir):
        for file in files:
            if file.endswith('.py') and file != '__init__.py':
                files_scanned += 1
                file_path = os.path.join(root, file)
                
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                
                for line_num, line in enumerate(lines, 1):
                    clean_line = line.strip()
                    
                    # 1. Skip comments and imports
                    if not clean_line or clean_line.startswith('#') or clean_line.startswith('import'):
                        continue
                        
                    for func in dangerous_functions:
                        pattern = rf'\.{func}\('
                        if re.search(pattern, clean_line):
                            
                            # 2. Check if it's a Causal Prefix (Rolling/EWM)
                            is_causal = any(re.search(safe, clean_line) for safe in safety_prefixes)
                            
                            # 3. Check if it's a Row-Wise operation (axis=1)
                            # This is the fix for the false positives you encountered
                            is_row_wise = any(marker in clean_line.replace(" ", "") for marker in row_wise_markers)
                            
                            if not is_causal and not is_row_wise:
                                print(f"🚨 POTENTIAL LEAK FOUND in {file} (Line {line_num}):")
                                print(f"   Code  : {clean_line}")
                                print(f"   Reason: Naked .{func}() detected. Not Rolling or Row-Wise.")
                                print("-" * 40)
                                violation_count += 1

    print(f"📊 SCAN COMPLETE:")
    print(f"   Files Scanned: {files_scanned}")
    if violation_count == 0:
        print("✅ PASS: Spectrally Clean. All logic is Causal or Row-Wise.")
    else:
        print(f"❌ FAIL: {violation_count} potential logic leaks remain.")

if __name__ == "__main__":
    run_global_stat_scan()