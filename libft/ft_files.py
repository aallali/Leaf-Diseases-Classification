import os
import numpy as np
import shutil
import random


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
            newPath = os.path.join(
                root_path,
                item[0]
            )
            all_images = np.append(
                all_images,
                ft_scrap_images(newPath)
            )
        else:
            all_images = np.append(
                all_images,
                os.path.join(root_path, item[0])
            )
    return all_images


def ft_form_image_path(destination, name, suffix=None, extension=".JPG"):
    imagePath = f"{destination}/"
    imagePath += f"{name}"

    if suffix is not None:
        imagePath += f"_{suffix}"

    imagePath += f".{extension}"
    return imagePath


def ft_split_dataset(
        input_folder,
        output_train_folder,
        output_test_folder,
        split_percentage
        ):
    """
    Split the dataset into training and testing sets while preserving
    the folder structure.

    Parameters:
    - input_folder (str): original dataset path.
    - output_train_folder (str): train data split path
    - output_test_folder (str): validation data split path
    - split_percentage (float): percent. of validation datasetset(e.g., 0.2).

    Returns:
    None

    This function creates two new folders, 'output_train_folder' and
    'output_test_folder', to store the training and testing datasets,
    respectively. The original folder structure is preserved in both sets.

    Note:
    - If 'output_train_folder' or 'output_test_folder' already exists,
        their contents will be deleted.

    Example usage:
    ```python
    ft_split_dataset("parent_folder", "split_train", "split_test", 0.2)
    ```

    This example splits the dataset in the 'parent_folder',
    saving 80% of the data in 'split_train' and 20% in 'split_test'.
    """
    if os.path.exists(output_train_folder):
        shutil.rmtree(output_train_folder)
    if os.path.exists(output_test_folder):
        shutil.rmtree(output_test_folder)

    for root, dirs, files in os.walk(input_folder):

        # Calculate the number of files to include in the test set
        num_files = len(files)
        num_test_files = int(num_files * split_percentage)

        # Randomly select files for the test set
        test_files = random.sample(files, num_test_files)
        train_files = [file for file in files if file not in test_files]

        # Create corresponding directory structure in the output train folder
        output_train_root = os.path.join(
            output_train_folder,
            os.path.relpath(root, input_folder)
        )
        os.makedirs(output_train_root, exist_ok=True)

        # Copy selected files to the train set folder
        for file in train_files:
            src_path = os.path.join(root, file)
            dest_path = os.path.join(output_train_root, file)
            shutil.copy2(src_path, dest_path)

        # Create corresponding directory structure in the output test folder
        output_test_root = os.path.join(
            output_test_folder,
            os.path.relpath(root, input_folder)
        )
        os.makedirs(output_test_root, exist_ok=True)

        # Copy selected files to the test set folder
        for file in test_files:
            src_path = os.path.join(root, file)
            dest_path = os.path.join(output_test_root, file)
            shutil.copy2(src_path, dest_path)
