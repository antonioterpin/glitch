import argparse
import yaml

def load_configuration(config_path):
    """Load the configuration file."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def argument_parser():
    parser = argparse.ArgumentParser(description="A simple argument parser.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs.yaml",
        help="Path to the configuration file.",
    )

    args = parser.parse_args(args)

    return args

if __name__ == "__main__":
    args = argument_parser()

    # Load the configuration file
    config = load_configuration(args.config)

    if config is None:
        raise ValueError("Configuration file not found or empty.")
    
    