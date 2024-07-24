import subprocess
import argparse

def main():
    root_dir = "/Users/romerocruzsa/Workspace/aiml/final-project-dialect-dynamics/src/"
    parser = argparse.ArgumentParser(description='Run training and evaluation for ASR model.')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--eval', action='store_true', help='Evaluate the model')
    
    args = parser.parse_args()

    if args.train:
        print("Starting training...")
        subprocess.run(['python', root_dir+'train.py'])

    if args.eval:
        print("Starting evaluation...")
        subprocess.run(['python', root_dir+'eval.py'])

if __name__ == "__main__":
    main()