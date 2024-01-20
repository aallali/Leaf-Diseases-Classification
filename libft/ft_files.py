import os
import numpy as np


def ft_list_imgs_and_folders(path):
    result = []
    try:
        # Get the list of files and folders in the specified path
        items = os.listdir(path)
        # Iterate through each item
        for item in items:
            item_path = os.path.join(path, item)

            # Check if the item is a file or a folder
            if os.path.isfile(item_path) and item_path.endswith(".JPG"):
                result.append((item, "image"))
            elif os.path.isdir(item_path):
                result.append((item, "folder"))

    except FileNotFoundError:
        print(f"The specified path '{ path }' does not exist.")

    return result


def ft_remove_prefix(path, prefix):
    return path[len(prefix):] if path.startswith(prefix) else path


def ft_scrap_images(root_path):
    all_images = []
    items_list = ft_list_imgs_and_folders(root_path)
    # Print the resulting list of tuples

    for item in items_list:
        if item[1] == "folder":
            newPath = os.path.join(root_path, item[0])
            all_images = np.append(all_images, ft_scrap_images(newPath))
        else:
            all_images = np.append(all_images, os.path.join(root_path, item[0]))
    return all_images
