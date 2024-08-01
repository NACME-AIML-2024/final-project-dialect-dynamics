import argparse
import subprocess

# Parse arguments
parser = argparse.ArgumentParser(description='Train and evaluate ASR model')
parser.add_argument('--train', action='store_true', help='Run training')
parser.add_argument('--eval', action='store_true', help='Run evaluation')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training and evaluation')
parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning Rate for training')
parser.add_argument('--gamma', type=float, default=0.90, help='Gamma for learning rate scheduler')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
parser.add_argument('--subset', type=int, help='Subset size of the data')
args = parser.parse_args()

root_dir = "/Users/romerocruzsa/Workspace/aiml/final-project-dialect-dynamics/src/"

# Train
if args.train:
    train_command = [
        "python", root_dir+"train.py",
        "--batch_size", str(args.batch_size),
        "--learning_rate", str(args.learning_rate),
        "--gamma", str(args.gamma),
        "--epochs", str(args.epochs)
    ]
    if args.subset:
        train_command.extend(["--subset", str(args.subset)])
    subprocess.run(train_command)

# Evaluate
if args.eval:
    eval_command = [
        "python", root_dir+"eval.py",
        "--batch_size", str(args.batch_size)
    ]
    if args.subset:
        eval_command.extend(["--subset", str(args.subset)])
    subprocess.run(eval_command)
