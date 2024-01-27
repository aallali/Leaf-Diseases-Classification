#!/usr/bin/env python3

import importlib
import argparse
import os
import shutil
from libft import ImageAugmentor, ft_clone_folder

Distribution = importlib.import_module("000-Distribution")

# number of possible augmentations
AUGMENTATIONS_TOTAL = 6


def augment_and_save_single_image(image_path, size,  export_path, plot):
    augmentor = ImageAugmentor(image_path, export_path)
    # Apply some random augmentations
    augmentor.some_augmentations(size)
    # Save augmented images
    augmentor.save_images()
    if plot:
        augmentor.show_augmented_images()


def augment_and_save_images_in_directory(directory_path, size, export_path):
    image_files = [f for f in os.listdir(directory_path) if
                   f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    files_generated = 0
    for image_file in image_files:
        image_path = os.path.join(directory_path, image_file)
        augmentor = ImageAugmentor(image_path, export_path)
        num_augmentations = AUGMENTATIONS_TOTAL if\
            size - files_generated > AUGMENTATIONS_TOTAL else\
            size - files_generated

        # Apply some random augmentations
        augmentor.some_augmentations(num_augmentations)

        # Save augmented images
        augmentor.save_images()
        files_generated += num_augmentations


def ft_augmentation(input_path, size, export_location):
    if os.path.exists(export_location):
        shutil.rmtree(export_location)

    if os.path.isfile(input_path):
        # If the input is a file
        augment_and_save_single_image(input_path, size,  export_location,
                                      True)
    elif os.path.isdir(input_path):

        # If the input is a directory
        data = Distribution.ft_distribution(
            input_path.replace('\\', '/'),
            AUGMENTATIONS_TOTAL
        )
        total_augmentations = data['total_augmentation_to_balance']

        for leaf in data['folder_statistics']:
            print(leaf)
            folderName = leaf[0]
            totalImgs = leaf[1]
            size = total_augmentations - totalImgs
            path_to_folder = data['image_paths'][folderName]['path_to_folder']
            imagePath = ''
            files_generated = 0

            ft_clone_folder(
                path_to_folder,
                export_location + path_to_folder.split("/")[-1]
            )

            for image in data['image_paths'][folderName]['images']:
                imagePath = f"{path_to_folder}/{image}"
                num_augmentations = 0
                if size - files_generated >= AUGMENTATIONS_TOTAL:
                    num_augmentations = AUGMENTATIONS_TOTAL
                else:
                    num_augmentations = max(size - files_generated, 0)

                if (num_augmentations):
                    augment_and_save_single_image(imagePath, num_augmentations,
                                                  export_location, False)
                    files_generated += num_augmentations

        return True
    else:
        print(f"Error: {input_path} is not a valid file or directory.")
        return False


def main():
    parser = argparse.ArgumentParser(description="Image Augmentation Script")
    parser.add_argument("input_path", help="Path to the image or directory")

    parser.add_argument("-size", type=int, default=AUGMENTATIONS_TOTAL,
                        help="Number of augmentations to generate\
                        (default: 6 and max is number of images multiplied\
                            by 6)")
    parser.add_argument(
        "-l",
        "--export_location",
        type=str,
        default="./augmented_directory/",
        help="images export location (default: given image or\
                            directory path)"
    )
    args = parser.parse_args()

    input_path, size, export_location = [
        args.input_path,
        args.size,
        (args.export_location + "/").replace("//", "/")
    ]

    ft_augmentation(input_path, size, export_location)


if __name__ == "__main__":
    main()
