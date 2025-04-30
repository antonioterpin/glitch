"""Module to check the correctness of the dataset generation and visualize the data."""
import jax
import numpy as np
import os, sys, argparse
import matplotlib.pyplot as plt
from glitch.dataloader import create_dataloaders

parser = argparse.ArgumentParser(description="Visualise TransitionsDataset batches")
parser.add_argument(
    "--save-dir", "-o",
    type=str,
    default=None,
    help="Directory in which to save every figure automatically. "
         "If omitted, figures are only saved when you press 's'."
)
args = parser.parse_args()

output_dir = "./out/debug/dataset"
if args.save_dir:
    output_dir = args.save_dir
os.makedirs(output_dir, exist_ok=True)

# 1. Configuration
config = {
    "dataset": {
        "batch_size": 10,
        "val_size":   32,
        "test_size":  32,
        "dataset_size": 100,
    },
    "problem": {
        "n_states":  2,
        "n_robots":  10,
        "horizon":   1,
    },
}

# 2. Instantiate loaders
train_loader, _, _ = create_dataloaders(config)
desired_bs = config["dataset"]["batch_size"]

# 3. Visualisation loop
batch_idx = 0
while True:
    batch = train_loader[batch_idx]

    # 2a. Sanity-check batch size
    initial_states, final_states = batch

    if initial_states.p.shape[0] != desired_bs:
        raise ValueError(
            f"Batch {batch_idx}: expected size {desired_bs}, "
            f"got {initial_states.shape[0]}"
        )

    # 3a. Pick first element in batch (index 0)
    init0  = np.asarray(jax.tree_util.tree_map(lambda x: x[0], initial_states).p[0, ...])
    final0 = np.asarray(jax.tree_util.tree_map(lambda x: x[0], final_states).p[0, ...])
    n_robots = init0.shape[0]

    # 3b. Plot and connect with arrows
    fig, ax = plt.subplots()
    ax.scatter(init0[:, 0],  init0[:, 1],  label="start", color="tab:blue")
    ax.scatter(final0[:, 0], final0[:, 1], label="goal",  color="tab:red")

    for r in range(n_robots):
        dx, dy = (final0[r] - init0[r])
        ax.arrow(
            init0[r, 0], init0[r, 1],
            dx, dy,
            length_includes_head=True,
            head_width=0.02, head_length=0.03,
            linewidth=0.8,
        )

    ax.set_title(f"Batch {batch_idx} - first sample ({n_robots} robots)")
    ax.set_aspect("equal")
    ax.legend(loc="best")
    plt.tight_layout()
    
    try:
        plt.show(block=False)
    except Exception as e:
        print("Cannot display figure. Save it instead.")

    # 4. Wait for user input before advancing
    print(
        "\nOptions:  [ENTER] next  |  s save & next  |  q quit"
        "\n→ ", end="", flush=True
    )
    choice = input().strip().lower()

    if choice == "q":
        print("Exiting.")
        plt.close("all")
        sys.exit(0)

    if choice == "s":
        os.makedirs(output_dir, exist_ok=True)
        fname = os.path.join(output_dir, f"batch_{batch_idx:04d}.png")
        fig.savefig(fname, dpi=300)
        print(f"Saved figure to {fname}")

    batch_idx += 1
    
