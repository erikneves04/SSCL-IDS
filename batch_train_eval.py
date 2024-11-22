import subprocess
import os

#corruption_rates = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
corruption_rates = [0.2, 0.3, 0.5, 0.7, 0.8]

output_dir = "evaluation_results"

os.makedirs(output_dir, exist_ok=True)

for rate in corruption_rates:
    print(f"Starting training for corruption rate: {rate}")
    
    train_command = f"python3 train.py --corruption_rate {rate}"
    subprocess.run(train_command, shell=True, check=True)
    
    print(f"Training for corruption rate {rate} completed. Starting evaluation.")
    
    eval_command = "python3 evaluation.py"
    eval_output = subprocess.run(eval_command, shell=True, check=True, text=True, capture_output=True)
    
    output_file = os.path.join(output_dir, f"evaluation_rate_{rate}.txt")
    with open(output_file, "w") as f:
        f.write(eval_output.stdout)
    
    print(f"Evaluation completed for corruption rate {rate}. Output saved to {output_file}.")
