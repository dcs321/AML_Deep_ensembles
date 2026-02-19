# Run training and evaluation
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb', action='store_true', help='Use wandb for logging')

    args = parser.parse_args()


if __name__ == "__main__":
    main()