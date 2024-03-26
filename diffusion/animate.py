import os
from PIL import Image

def images_to_gif(folder_path, gif_path):
    images = []
    for i in range(0, 199, 4):
        filename = f'timestep_{i}.png'
        image_path = os.path.join(folder_path, filename)
        if os.path.exists(image_path):
            images.append(Image.open(image_path))

    images[0].save(gif_path, save_all=True, append_images=images[1:], duration=100, loop=0)

if __name__ == "__main__":
    folder_path = 'figures'  # Folder containing PNG images
    gif_path = 'backward.gif'  # Path to save the GIF
    images_to_gif(folder_path, gif_path)
