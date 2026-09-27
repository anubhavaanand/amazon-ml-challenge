import os
import sys
import subprocess

print("Cloning repository...")
subprocess.run("git clone https://github.com/anubhavaanand/entity-resolve-ml.git", shell=True, check=True)

os.chdir("entity-resolve-ml")

print("Installing dependencies...")
subprocess.run("pip install rapidfuzz scikit-learn tqdm", shell=True, check=True)

# Dynamically locate the dataset folder containing train and test
print("Searching for dataset directory in /kaggle/input...")
DATA_DIR = None
for root, dirs, files in os.walk("/kaggle/input"):
    if "train_source2.tsv" in files:
        DATA_DIR = os.path.dirname(root)
        print(f"Found dataset directory at: {DATA_DIR}")
        break

if not DATA_DIR:
    DATA_DIR = "/kaggle/input/datasets/anubhavaanand/amazon-ml-challenge/cp/dataset"
    print(f"Fallback dataset directory: {DATA_DIR}")

print(f"Listing contents of {DATA_DIR}:")
try:
    print(os.listdir(DATA_DIR))
except Exception as e:
    print(f"Could not list {DATA_DIR}: {e}")

# 1. Train Split Blocking
print("Starting V2 Blocking on TRAIN split...")
cmd_train = f"python src/blocking/blocking_v2.py --data_dir {DATA_DIR}/train --split train --out_file /kaggle/working/v2_train_candidates.tsv --mode two_stage"
print(f"Running: {cmd_train}")
subprocess.run(cmd_train, shell=True, check=True)

# 2. Test Split Blocking
print("Starting V2 Blocking on TEST split...")
cmd_test = f"python src/blocking/blocking_v2.py --data_dir {DATA_DIR}/test --split test --out_file /kaggle/working/v2_test_candidates.tsv --mode two_stage"
print(f"Running: {cmd_test}")
subprocess.run(cmd_test, shell=True, check=True)

print("V2 Blocking Complete! Files saved to /kaggle/working/")
