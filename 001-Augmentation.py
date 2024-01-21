# write augmentation logic here
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import sys
import importlib  
Distribution = importlib.import_module("000-Distribution")

ft_distribution = Distribution.ft_distribution
# number of possible augmentations
augmentations_total = 7
class ImageAugmentor:
    def __init__(self, image_path, export_folder):
        """
        Initializes the ImageAugmentor object.

        Args:
        - image_path (str): The path to the input image.
        """
        self.image_path = image_path.replace('\\', '/')
        self.image_name = self.get_image_name(self.image_path)
        self.directory = os.path.dirname(self.image_path) 
        self.export_directory = os.path.dirname(self.image_path) if export_folder is None else export_folder
        self.image_extension = self.get_image_extension(self.image_path) 
        self.image = cv2.imread(self.image_path)
        self.augmented_images = []  # Initialize with the original image
        
    def get_image_name(self, image_path):
        """
        Extracts the image name from the path.

        Args:
        - image_path (str): The path to the input image.

        Returns:
        - str: The extracted image name.
        """
        return image_path.split('/')[-1].split('.')[0]
    
    def get_image_extension(self, image_path):
        """
        Extracts the image extension from the path.

        Args:
        - image_path (str): The path to the input image.

        Returns:
        - str: The extracted image extension.
        """
        return image_path.split('/')[-1].split('.')[-1]
    
    def flip_vertical(self):
        """
        Applies vertical flipping to the image and appends the result to augmented_images.
        """
        flipped_image = cv2.flip(self.image, 0)
        self.augmented_images.append({'name': 'Flip', 'image': flipped_image})
        
    def rotate(self, angle):
        """
        Rotates the image by the specified angle and appends the result to augmented_images.

        Args:
        - angle (float): The angle of rotation.
        """
        img_height, img_width = self.image.shape[0], self.image.shape[1]
        centre_y, centre_x = img_height // 2, img_width // 2

        rotation_matrix = cv2.getRotationMatrix2D((centre_y, centre_x), angle, 1.0)
        cos_of_rotation_matrix = np.abs(rotation_matrix[0][0])
        sin_of_rotation_matrix = np.abs(rotation_matrix[0][1])

        new_image_height = int((img_height * sin_of_rotation_matrix) + (img_width * cos_of_rotation_matrix))
        new_image_width = int((img_height * cos_of_rotation_matrix) + (img_width * sin_of_rotation_matrix))

        rotation_matrix[0][2] += (new_image_width/2) - centre_x
        rotation_matrix[1][2] += (new_image_height/2) - centre_y

        rotating_image = cv2.warpAffine(self.image, rotation_matrix, (new_image_width, new_image_height))
        self.augmented_images.append({'name': 'Rotation', 'image': rotating_image})

    def scale(self, scale_factor):
        """
        Scales the image by the specified factor and appends the result to augmented_images.

        Args:
        - scale_factor (float): The scaling factor.
        """
        rows, cols, _ = self.image.shape
        zoomed_image = cv2.resize(self.image[rows//4:3*rows//4, cols//4:3*cols//4], (cols, rows))
        self.augmented_images.append({'name': 'Scaling', 'image': zoomed_image})

    def blur(self):
        """
        Applies Gaussian blur to the image and appends the result to augmented_images.
        """
        blurred_image = cv2.GaussianBlur(self.image, (15, 15), 0)
        self.augmented_images.append({'name': 'Blur', 'image': blurred_image})

    def adjust_contrast(self, contrast_factor):
        """
        Adjusts the contrast of the image and appends the result to augmented_images.

        Args:
        - contrast_factor (float): The contrast adjustment factor.
        """
        adjusted_image = cv2.convertScaleAbs(self.image, alpha=contrast_factor, beta=2)
        self.augmented_images.append({'name': 'Contrast', 'image': adjusted_image})

    def brightness(self, brightness_factor):
        """
        Adjusts the brightness of the image and appends the result to augmented_images.

        Args:
        - brightness_factor (float): The brightness adjustment factor.
        """
        adjusted_image = cv2.convertScaleAbs(self.image, beta=brightness_factor)
        self.augmented_images.append({'name': 'Brightness', 'image': adjusted_image})

    def projection(self, perspective_factor):
        """
        Applies a perspective transformation to the image and appends the result to augmented_images.

        Args:
        - perspective_factor (float): The perspective transformation factor.
        """
        rows, cols, _ = self.image.shape
        src_points = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1], [cols - 1, rows - 1]])
        dst_points = np.float32([
            [int(0.1 * cols), int(0.3 * rows)],
            [int(0.9 * cols), int(0.1 * rows)],
            [int(0.3 * cols), int(0.9 * rows)],
            [int(0.9 * cols), int(0.6 * rows)]
        ])
        perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
                # Apply perspective transformation
        perspective_transformed_image = cv2.warpPerspective(self.image, perspective_matrix, (cols, rows), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        self.augmented_images.append({'name':'Pojection', 'image':perspective_transformed_image})
    

    def show_augmented_images(self):
        num_images = len(self.augmented_images)

        fig, axes = plt.subplots(1, num_images, figsize=(15, 5))

        for i in range(num_images):
            axes[i].imshow(cv2.cvtColor(self.augmented_images[i]['image'], cv2.COLOR_BGR2RGB))
            axes[i].axis('off')
            axes[i].set_title(f"{self.augmented_images[i]['name']}")

        plt.show()
        
    def save_images(self):
        for image in self.augmented_images:
            
            # imagePath = f"{self.image_name}_{image['name']}.{self.image_extenssion}"
            imagePath = f"{self.export_directory}/{self.image_name}_{image['name']}.{self.image_extension}"
            if not os.path.exists(self.export_directory):
                # If it doesn't exist, create it
                os.makedirs(self.export_directory)
            cv2.imwrite(imagePath, image['image'])
            
    def some_augmentations(self, num_operations):
        available_operations = [
            self.rotate,
            self.blur,
            self.adjust_contrast,
            self.scale,
            self.brightness,
            self.projection,
            self.flip_vertical,
            # Add other augmentation methods as needed
        ]
        # Randomize the list of operations
        random.shuffle(available_operations)

        # Apply the selected number of operations
        for operation in available_operations[:max(num_operations , 0)]:
            # Generate random parameters if needed for each operation
            if operation.__name__ == 'rotate':
                angle = random.uniform(-45, 45)
                operation(angle)
            elif operation.__name__ == 'blur':
                operation()
            elif operation.__name__ == 'scale':
                operation(2)
            elif operation.__name__ == 'brightness':
                operation(60)
            elif operation.__name__ == 'projection':
                operation(0.8)
            elif operation.__name__ == 'flip_vertical':
                operation()
            elif operation.__name__ == 'adjust_contrast':
                factor = random.uniform(0.5, 2.0)
                operation(factor)


import sys
import argparse


 



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
            export_path = f"./test/{leaf}"
            imagePath = ''
            files_generated = 0
            
            for image in data['image_paths'][leaf]['images']:
                imagePath = f"{path_to_folder}/{image}"
                num_augmentations = augmentations_total if size - files_generated > augmentations_total else max(size - files_generated,0)
                files_generated += num_augmentations
                augment_and_save_single_image(imagePath, num_augmentations, export_path)
            
    else:
        print(f"Error: {input_path} is not a valid file or directory.")
        sys.exit(1)
def main():
    parser = argparse.ArgumentParser(description="Image Augmentation Script")
    parser.add_argument("input_path", help="Path to the image or directory")
    parser.add_argument("-size", type=int, default=augmentations_total, help="Number of augmentations to generate (default: 7 and max is number of images multiplied by 7)")
    parser.add_argument("-export_location", type=str, default=None, help="images export location (default: image or directory path)")
    args = parser.parse_args()
    input_path, size, export_location = [args.input_path, args.size, args.export_location]
    
    ft_augmentation(input_path, size, export_location)
    
    


def augment_and_save_single_image(image_path, size,  export_path):
    augmentor = ImageAugmentor(image_path, export_path)
    for i in range(size):
        num_augmentations = augmentations_total
        # Apply some random augmentations
        augmentor.some_augmentations(num_augmentations)
        
        # Save augmented images
        augmentor.save_images()

def augment_and_save_images_in_directory(directory_path, size, export_path):
    image_files = [f for f in os.listdir(directory_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    files_generated = 0
    for image_file in image_files:
        image_path = os.path.join(directory_path, image_file)
        augmentor = ImageAugmentor(image_path, export_path)

        
        num_augmentations = augmentations_total if size - files_generated > augmentations_total else size - files_generated

        # Apply some random augmentations
        augmentor.some_augmentations(num_augmentations)
        
        # Save augmented images
        augmentor.save_images()
        files_generated += num_augmentations

if __name__ == "__main__":
    main()
