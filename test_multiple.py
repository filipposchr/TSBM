import argparse
import subprocess

datasets = [ "edit-mathoverflow", "edit-facebook_wall", "edit-topology",  "edit-mlwikiquote", "edit-plwikiquote", "edit-digg_reply", "edit-SMS", "edit-rt-pol", "edit-slashdot_reply","edit-wamazon", "edit-mgwikipedia", "edit-tgwiktionary", "edit-ltwiktionary"]

parser = argparse.ArgumentParser()
parser.add_argument("--bet", choices=["sfm", "sh"], required=True, help="Choose between 'sfm' or 'sh'")
args = parser.parse_args()

# Loop over each dataset
for d in datasets:
    print(f"\n - Running evaluation for dataset: *{d}* -")

    # Construct command
    command = ["python", "-u", "main.py", "-d", d, "--bet", args.bet, "--test"]

    # Run the command
    result = subprocess.run(command)

    # Optional: Check if it ran successfully
    if result.returncode != 0:
        print(f"Error: Evaluation failed for dataset {d}")
