import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

# Load the feature maps from the pickle file
name = "featuremaps_unet"
with open(f'{name}.pkl', 'rb') as f:
    feature_maps = torch.load(f)

# Define a directory to save the visualizations
output_dir = f'{name}_visualizations'
os.makedirs(output_dir, exist_ok=True)

# Precompute the global min and max values for colormap normalization
global_min, global_max = float('inf'), float('-inf')

# First pass to find the global min and max across all feature maps
for layer_name, fmap in feature_maps:
    if isinstance(fmap, dict):
        fmap = fmap['xy']  # or whichever key is relevant in your context
    fmap = fmap.squeeze(0)  # Remove the batch dimension
    fmap_flat = fmap.view(fmap.shape[0], -1).cpu().numpy().T
    pca = PCA(n_components=1)
    fmap_pca = pca.fit_transform(fmap_flat).reshape(fmap.shape[1], fmap.shape[2])
    global_min = min(global_min, fmap_pca.min())
    global_max = max(global_max, fmap_pca.max())

# Second pass to plot with normalized colormap
for layer_name, fmap in feature_maps:
    if isinstance(fmap, dict):
        fmap = fmap['xy']  # or whichever key is relevant in your context
    fmap = fmap.squeeze(0)  # Remove the batch dimension
    fmap_flat = fmap.view(fmap.shape[0], -1).cpu().numpy().T
    pca = PCA(n_components=1)
    fmap_pca = pca.fit_transform(fmap_flat).reshape(fmap.shape[1], fmap.shape[2])

    # Plot the feature map using Turbo colormap
    plt.imshow(fmap_pca, cmap='turbo', vmin=global_min, vmax=global_max)
    plt.axis('off')  # Hide axes
    plt.gca().set_frame_on(False)  # Remove frame around the image

    # Save the image
    file_name = os.path.join(output_dir, f'{layer_name}_pca.png')
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    plt.close()

print(f"Feature maps saved in '{output_dir}' directory.")
