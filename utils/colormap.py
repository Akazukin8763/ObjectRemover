import numpy as np
import matplotlib.pyplot as plt


def generate_colors(num_colors):
    # Use the hsv color map to generate evenly spaced colors
    hsv_colors = np.linspace(0, 1, num_colors)
    cmap = plt.get_cmap('hsv')

    # Convert HSV colors to RGB
    rgb_colors = cmap(hsv_colors)[:, :3]

    # Convert to 0-255 range and integer type for OpenCV compatibility
    rgb_colors = (rgb_colors * 255).astype(int)

    return rgb_colors


# Generate a specific number of colors
_NUM_COLORS = 60
_COLORS = generate_colors(num_colors=_NUM_COLORS)
_COLORS = _COLORS.astype(np.float32).reshape(-1, 3)


def random_color():
    idx = np.random.randint(0, _NUM_COLORS)
    return _COLORS[idx]


def random_colors(size: int):
    indices = np.random.choice(_NUM_COLORS, size, replace=False)
    return _COLORS[indices]
