import os
import subprocess
import fire

def main(target_dir:str):
    print(f"Submitting scripts in {target_dir}")
    for file in os.listdir(target_dir):
        if file.endswith(".bash"):
            subprocess.run(["sbatch", f"{target_dir}/{file}"])
            print(f"Submitted {file}")

if __name__ == "__main__":
    fire.Fire(main)

