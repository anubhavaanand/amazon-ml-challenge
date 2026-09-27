import os
import subprocess

print("Cloning repository...")
subprocess.run("git clone https://github.com/anubhavaanand/entity-resolve-ml.git", shell=True, check=True)

os.chdir("entity-resolve-ml")

print("Installing dependencies...")
subprocess.run("pip install rapidfuzz scikit-learn tqdm", shell=True, check=True)

# Define data directory from Kaggle datasets
DATA_DIR = "/kaggle/input/amazon-ml-challenge/dataset"

# 1. Train Split Blocking
print("Starting V2 Blocking on TRAIN split...")
subprocess.run(f"python src/blocking/blocking_v2.py --data_dir {DATA_DIR}/train --split train --out_file /kaggle/working/v2_train_candidates.tsv --mode two_stage", shell=True, check=True)

# 2. Test Split Blocking
print("Starting V2 Blocking on TEST split...")
subprocess.run(f"python src/blocking/blocking_v2.py --data_dir {DATA_DIR}/test --split test --out_file /kaggle/working/v2_test_candidates.tsv --mode two_stage", shell=True, check=True)

print("V2 Blocking Complete!")
