import subprocess
import itertools
import sys
import time

# Define the parameter combinations
models = ['--GMIX25', '--VIT20', '--ENET20', '--ENET4', '--VIT80']
options = ['', '--quant', '--onnx', '--onnx --onnx-reparam', '--cpu']
hardware = ['', '--quadro']

# Generate all combinations of parameters
combinations = itertools.product(models, options, hardware)

# Path to the script to be tested
script_to_test = 'python run.py'

def test_combinations(extra_args):
    for combo in combinations:
        model, option, hw = combo
        # Build the command to execute with extra CLI arguments
        command = f'{script_to_test} {" ".join(extra_args)} {model} {option} {hw}'
        print(f"Running: {command}")

        # Measure execution time
        start_time = time.time()

        # Run the script and capture output
        try:
            result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Output:\n{result.stdout.decode()}")
            print(f"Execution time: {elapsed_time:.2f} seconds")
        except subprocess.CalledProcessError as e:
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Error while running {command}:\n{e.stderr.decode()}")
            print(f"Execution time: {elapsed_time:.2f} seconds")

if __name__ == '__main__':
    # Pass all arguments after tester.py to run.py
    extra_args = sys.argv[1:]
    test_combinations(extra_args)
