import matplotlib.pyplot as plt


def plot_features_extraction(c_gt, c_rec, descriptor_name="", figsize=(10, 5)):
    # Create figure and axis
    f, ax = plt.subplots(1, figsize=figsize)

    # Convert tensors to NumPy arrays
    gt = c_gt.detach().cpu().numpy()
    rec = None
    if c_rec is not None:
        rec = c_rec.detach().cpu().numpy()
    
    # Plot each line with label
    ax.plot(gt, label='Ground Truth', color='C0', linewidth=2)
    if c_rec is not None:
        ax.plot(rec, label='Ours', color='C1', linestyle='--', linewidth=2, alpha=0.9)
    
    # Set title and labels
    ax.set_title(descriptor_name, fontsize=14, weight='bold')
    ax.set_xlabel('latent frames', fontsize=12)
    ax.set_ylabel('control variable', fontsize=12)

    # Set y-axis limits
    ax.set_ylim(-1.2, 1.2)

    # Add legend and grid
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.7)

    # Improve layout
    plt.tight_layout()
    return f
