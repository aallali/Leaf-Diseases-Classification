# write augmentation logic here
import importlib
import argparse
import os
from libft import ImageAugmentor

Distribution = importlib.import_module("000-Distribution")

# number of possible augmentations
AUGMENTATIONS_TOTAL = 7


def augment_and_save_single_image(image_path, size,  export_path):
    augmentor = ImageAugmentor(image_path, export_path)
    # Apply some random augmentations
    augmentor.some_augmentations(size)
    # print("augmented images", len(augmentor.augmented_images))
    # print(size)
    # Save augmented images
    augmentor.save_images()


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
    if os.path.isfile(input_path):
        # If the input is a file
        augment_and_save_single_image(input_path, size,  export_location)
    elif os.path.isdir(input_path):
        # If the input is a directory
        data = Distribution.ft_distribution(input_path.replace('\\', '/'), 7)
        total_augmentations = data['total_augmentation_to_balance']

        for leaf in data['folder_statistics']:
            print(leaf)
            size = total_augmentations - data['folder_statistics'][leaf]
            path_to_folder = data['image_paths'][leaf]['path_to_folder']
            # export_path = f"{path_to_folder}"
            # export_path = f"./test/{leaf}" if export_location else\
            #     f"{path_to_folder}"
            imagePath = ''
            files_generated = 0
            for image in data['image_paths'][leaf]['images']:
                imagePath = f"{path_to_folder}/{image}"
                num_augmentations = AUGMENTATIONS_TOTAL if\
                    size - files_generated >= AUGMENTATIONS_TOTAL else\
                    max(size - files_generated, 0)
                files_generated += num_augmentations
                if (num_augmentations):
                    augment_and_save_single_image(imagePath, num_augmentations,
                                                  export_location)
        return True
    else:
        print(f"Error: {input_path} is not a valid file or directory.")
        return False


def main():
    parser = argparse.ArgumentParser(description="Image Augmentation Script")
    parser.add_argument("input_path", help="Path to the image or directory")
    parser.add_argument("-size", type=int, default=AUGMENTATIONS_TOTAL,
                        help="Number of augmentations to generate\
                        (default: 7 and max is number of images multiplied\
                            by 7)")
    parser.add_argument("-export_location", type=str, default=None,
                        help="images export location (default: given image or\
                            directory path)")
    args = parser.parse_args()
    input_path, size, export_location = [args.input_path, args.size,
                                         args.export_location]

    ft_augmentation(input_path, size, export_location)


if __name__ == "__main__":
    main()
