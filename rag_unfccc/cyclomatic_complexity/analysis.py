#!/usr/bin/env python3
"""
Cyclomatic Complexity Analysis Script
Usage: python analysis.py [file_or_directory]
If no path specified, analyzes current directory
"""

import os
import sys
import json
import glob
import subprocess
import datetime
import platform
from pathlib import Path

def command_exists(command):
    """Check if a command exists by running 'which' or 'where'."""
    if platform.system() == "Windows":
        cmd = ["where", command]
    else:
        cmd = ["which", command]
    
    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except subprocess.CalledProcessError:
        return False

def install_dependencies():
    """Install required dependencies if they don't exist."""
    dependencies = ["radon", "flake8", "pylint"]
    for dep in dependencies:
        try:
            if not command_exists(dep):
                print(f"Installing {dep}...")
                subprocess.run([sys.executable, "-m", "pip", "install", dep], check=True)
            else:
                print(f"✓ {dep} already installed")
        except subprocess.CalledProcessError as e:
            print(f"Error installing {dep}: {str(e)}")
            sys.exit(1)

def run_radon_analysis(target_path, report_dir):
    """Run Radon analysis on the target path."""
    print("\n🎯 RADON ANALYSIS - Cyclomatic Complexity")
    print("=" * 50)
    
    print("\n📈 High Complexity Functions (Grade C and below):")
    high_complexity_file = os.path.join(report_dir, "radon_high_complexity.txt")
    with open(high_complexity_file, "w") as f:
        subprocess.run(["radon", "cc", target_path, "--min", "C", "--show-complexity", "--total-average"], 
                      stdout=f, stderr=subprocess.STDOUT)
    
    # Display the content
    with open(high_complexity_file, "r") as f:
        print(f.read())
    
    print("\n📊 All Functions Summary:")
    all_functions_file = os.path.join(report_dir, "radon_all_functions.txt")
    with open(all_functions_file, "w") as f:
        subprocess.run(["radon", "cc", target_path, "--total-average"], 
                      stdout=f, stderr=subprocess.STDOUT)
    
    # Display first 20 lines
    with open(all_functions_file, "r") as f:
        lines = f.readlines()
        print("".join(lines[:20]))
    
    print("\n🔥 Top 10 Most Complex Functions:")
    json_file = os.path.join(report_dir, "radon_json.json")
    with open(json_file, "w") as f:
        subprocess.run(["radon", "cc", target_path, "--order", "SCORE", "--json"], 
                      stdout=f, stderr=subprocess.STDOUT)
    
    try:
        with open(json_file, "r") as f:
            data = json.load(f)
        
        functions = []
        for file_path, file_data in data.items():
            for item in file_data:
                if item['type'] == 'function' or item['type'] == 'method':
                    functions.append({
                        'name': item['name'],
                        'file': file_path,
                        'complexity': item['complexity'],
                        'rank': item['rank'],
                        'lineno': item['lineno']
                    })
        
        # Sort by complexity
        functions.sort(key=lambda x: x['complexity'], reverse=True)
        
        print('Rank | Complexity | Function | File:Line')
        print('-' * 60)
        for i, func in enumerate(functions[:10], 1):
            print(f"{func['rank']:4} | {func['complexity']:10} | {func['name']:20} | {func['file']}:{func['lineno']}")
            
    except Exception as e:
        print(f'Error processing JSON: {e}')

def run_flake8_analysis(target_path, report_dir):
    """Run Flake8 analysis on the target path."""
    print("\n🎯 FLAKE8 ANALYSIS - General Code Quality + Complexity")
    print("=" * 50)
    
    print("\n⚠️  Complexity Issues (> 10):")
    complexity_file = os.path.join(report_dir, "flake8_complexity.txt")
    
    # Run flake8 with complexity check
    subprocess.run(["flake8", target_path, "--max-complexity=10", "--statistics", 
                   "--output-file", complexity_file], 
                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    
    # Display complexity issues
    with open(complexity_file, "r") as f:
        content = f.read()
        if "C901" in content:
            print(content)
        else:
            print("No high complexity issues found!")
    
    print("\n📋 General Code Quality Issues:")
    all_issues_file = os.path.join(report_dir, "flake8_all_issues.txt")
    
    # Run flake8 for all issues
    subprocess.run(["flake8", target_path, "--statistics", "--output-file", all_issues_file], 
                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    
    # Display last 20 lines
    with open(all_issues_file, "r") as f:
        lines = f.readlines()
        print("".join(lines[-20:]))

def run_pylint_analysis(target_path, report_dir):
    """Run Pylint analysis on the target path."""
    print("\n🎯 PYLINT ANALYSIS - Comprehensive Code Analysis")
    print("=" * 50)
    
    print("🔍 Running Pylint (this may take a moment)...")
    pylint_file = os.path.join(report_dir, "pylint_complexity.txt")
    
    # Get list of Python files to analyze
    if os.path.isdir(target_path):
        python_files = glob.glob(os.path.join(target_path, "**/*.py"), recursive=True)
        # Limit to first 5 files for faster execution
        python_files = python_files[:5]
    else:
        python_files = [target_path]
    
    with open(pylint_file, "w") as f:
        for py_file in python_files:
            f.write(f"Analyzing: {py_file}\n")
            result = subprocess.run(["pylint", py_file, "--disable=all", 
                                    "--enable=too-many-branches,too-many-statements,too-complex"],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            f.write(result.stdout)
    
    # Display content if not empty
    with open(pylint_file, "r") as f:
        content = f.read()
        if len(content.strip()) > 0:
            print(content)
        else:
            print("✅ No major complexity issues found by Pylint!")

def generate_summary(target_path, report_dir):
    """Generate a summary report of the analysis."""
    print("\n📋 SUMMARY & RECOMMENDATIONS")
    print("=" * 50)
    
    print('🎯 KEY FINDINGS:')
    print()
    
    # Count Python files
    if os.path.isdir(target_path):
        py_files = len(glob.glob(os.path.join(target_path, "**/*.py"), recursive=True))
        print(f'📁 Python files analyzed: {py_files}')
    else:
        print(f'📄 Python file analyzed: {os.path.basename(target_path)}')
    
    # Check radon results
    radon_file = os.path.join(report_dir, "radon_high_complexity.txt")
    if os.path.exists(radon_file):
        with open(radon_file, 'r') as f:
            content = f.read()
            complex_functions = content.count('def ') + content.count('class ')
            if complex_functions > 0:
                print(f'⚠️  High complexity functions/classes: {complex_functions}')
            else:
                print('✅ No high complexity functions found')
    
    # Check flake8 results
    flake8_file = os.path.join(report_dir, "flake8_complexity.txt")
    complexity_issues = 0
    if os.path.exists(flake8_file):
        with open(flake8_file, 'r') as f:
            content = f.read()
            complexity_issues = content.count('C901')
    
    if complexity_issues > 0:
        print(f'⚠️  Flake8 complexity violations: {complexity_issues}')
    else:
        print('✅ No flake8 complexity violations')
    
    print()
    print('🔧 RECOMMENDED ACTIONS:')
    print('1. Focus on functions with complexity > 15 first')
    print('2. Use early returns to reduce nesting')
    print('3. Extract complex conditions into named variables')
    print('4. Break large functions into smaller ones')
    print('5. Consider using polymorphism for complex switch/if chains')
    print()
    print(f'📊 Detailed reports saved in: {report_dir}/')
    print('   - radon_high_complexity.txt: Functions to refactor')
    print('   - flake8_complexity.txt: Complexity violations')
    print('   - pylint_complexity.txt: Comprehensive analysis')

def main():
    """Main function to run the complexity analysis."""
    # Get target path from command line argument or use current directory
    target_path = sys.argv[1] if len(sys.argv) > 1 else "."
    
    # Handle relative paths to parent directory
    if target_path.startswith("..") or target_path == "..":
        # Converting to absolute path relative to script's parent directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        
        if target_path == "..":
            target_path = parent_dir
        else:
            # For paths like "../something"
            rel_path = target_path[3:] if target_path.startswith("../") else target_path[2:]
            target_path = os.path.join(parent_dir, rel_path)
    
    # Validate the path
    if not os.path.exists(target_path):
        print(f"Error: Path '{target_path}' does not exist.")
        sys.exit(1)
    
    # Determine if target is a file or directory
    is_file = os.path.isfile(target_path)
    
    # Check if it's a Python file when a file is specified
    if is_file and not target_path.endswith('.py'):
        print(f"Error: '{target_path}' is not a Python file. Please specify a .py file or a directory.")
        sys.exit(1)
    
    # Create timestamp for report directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = f"complexity_reports_{timestamp}"
    os.makedirs(report_dir, exist_ok=True)
    
    # Print analysis info
    print("🔍 Starting Cyclomatic Complexity Analysis...")
    if is_file:
        print(f"📄 Target File: {target_path}")
    else:
        print(f"📁 Target Directory: {target_path}")
    print(f"📊 Report Directory: {report_dir}")
    print("=" * 60)
    
    # Check and install dependencies
    print("🔧 Checking dependencies...")
    install_dependencies()
    print("✅ Dependencies ready!")
    
    # Run analysis tools
    run_radon_analysis(target_path, report_dir)
    run_flake8_analysis(target_path, report_dir)
    run_pylint_analysis(target_path, report_dir)
    
    # Generate summary
    generate_summary(target_path, report_dir)
    
    print("\n🏁 Analysis Complete!")
    print(f"📂 Check the '{report_dir}' directory for detailed reports.")

if __name__ == "__main__":
    main()
