import os
import datetime
import matplotlib.pyplot as plt

def plot_heatmaps(data, output_folder, num_heads=4, title_font_size=12, subtitle_font_size=10, figsize=(120, 40), dpi=100,
                  row_spacing=0.2, column_spacing=0.2, title_spacing=0.4, subtitle_spacing=0.3, tick_font_size=8):
    # Create the output folder if it doesn't exist

    output_path = os.path.join(os.environ.get('HOME'), 'Code', 'juan-garassino', 'mySandbox', 'nanoGPT', 'plots', output_folder)

    os.makedirs(output_path, exist_ok=True)

    for number, key in enumerate(data.keys()):
        print(number +1, key)

    # Retrieve the keys of the linear layers
    linear_layer_keys = [key for key in data.keys() if key.endswith('.weight')]

    # Calculate the number of linear layers and heads

    num_layers = len(linear_layer_keys) // 4

    print(num_layers)

    print(num_heads)

    # Create the subplot grid
    fig, axes = plt.subplots(num_layers, num_heads, figsize=figsize, dpi=dpi)

    # Adjust the row and column spacing
    fig.subplots_adjust(hspace=row_spacing, wspace=column_spacing, top=1 - title_spacing / figsize[1])

    # Iterate over the linear layers and plot the heatmaps
    for i, layer_key in enumerate(linear_layer_keys):
        # Extract the weights from the layer
        weights = data[layer_key]

        # Move the weights to CPU memory
        weights = weights.cpu()

        # Reshape the weights to 2D
        weights_2d = weights.view(weights.size(0), -1)

        # Convert the weights to a numpy array
        weights_np = weights_2d.detach().numpy()

        # Determine the layer and head indices
        layer_idx = i % num_layers
        head_idx = i // num_layers

        # Plot the heatmap in the corresponding subplot
        axes[layer_idx, head_idx].imshow(weights_np, cmap='hot')
        axes[layer_idx, head_idx].set_title(f"Layer: {layer_idx}\nHead: {head_idx+1}", fontsize=subtitle_font_size,
                                            pad=subtitle_spacing)

        # Set the font size of the ticks
        axes[layer_idx, head_idx].tick_params(axis='both', which='both', labelsize=tick_font_size)

    # Adjust the layout and spacing
    fig.tight_layout()

    # Save the plot with timestamp in the output folder
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    output_path = os.path.join(os.environ.get('HOME'), 'Code', 'juan-garassino', 'mySandbox', 'nanoGPT', 'plots', output_folder, f'heatmaps[{timestamp}].png')
    plt.savefig(output_path, dpi=dpi)
    plt.close(fig)

    print(f"Saved heatmaps at: {output_path}")
