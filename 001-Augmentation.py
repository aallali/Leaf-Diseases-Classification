# write augmentation logic here
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import sys
# number of possible augmentations

class ImageAugmentor:
    def __init__(self, image_path):
        self.image_name = self.get_image_name(image_path)
        self.directory  = os.path.dirname(image_path)
        self.image_extenssion = self.get_image_extenssion(image_path)
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        self.augmented_images = []  # Initialize with the original image
        
    def get_image_name(self, image_path):
        # Extract the image name from the path (you may need to adjust this based on your file naming conventions)
        return image_path.split('/')[-1].split('.')[0]
    
    def get_image_extenssion(self, image_path):
        # Extract the image name from the path (you may need to adjust this based on your file naming conventions)
        return image_path.split('/')[-1].split('.')[-1]
    
    def flip_vertical(self):
        flipped_image = cv2.flip(self.image, 0)
        self.augmented_images.append(flipped_image)
        
    def rotate(self, angle):
        # Taking image height and width 
        imgHeight, imgWidth = self.image.shape[0], self.image.shape[1] 
    
        # Computing the centre x,y coordinates 
        # of an image 
        centreY, centreX = imgHeight//2, imgWidth//2
    
        # Computing 2D rotation Matrix to rotate an image 
        rotationMatrix = cv2.getRotationMatrix2D((centreY, centreX), angle, 1.0) 
    
        # Now will take out sin and cos values from rotationMatrix 
        # Also used numpy absolute function to make positive value 
        cosofRotationMatrix = np.abs(rotationMatrix[0][0]) 
        sinofRotationMatrix = np.abs(rotationMatrix[0][1]) 
    
        # Now will compute new height & width of 
        # an image so that we can use it in 
        # warpAffine function to prevent cropping of image sides 
        newImageHeight = int((imgHeight * sinofRotationMatrix) +
                            (imgWidth * cosofRotationMatrix)) 
        newImageWidth = int((imgHeight * cosofRotationMatrix) +
                            (imgWidth * sinofRotationMatrix)) 
    
        # After computing the new height & width of an image 
        # we also need to update the values of rotation matrix 
        rotationMatrix[0][2] += (newImageWidth/2) - centreX 
        rotationMatrix[1][2] += (newImageHeight/2) - centreY 
    
        # Now, we will perform actual image rotation 
        rotatingimage = cv2.warpAffine( 
            self.image, rotationMatrix, (newImageWidth, newImageHeight)) 
        self.augmented_images.append({'name':'Rotation', 'image':rotatingimage})

    def scale(self, scale_factor):
        rows, cols, _ = self.image.shape
        # Calculate the zoom-in region and resize
        zoomed_image = cv2.resize(self.image[rows//4:3*rows//4, cols//4:3*cols//4], (cols, rows))
        self.augmented_images.append({'name':'Scaling', 'image': zoomed_image})

    def blur(self):
        """
        
        """
        blurred_image = cv2.GaussianBlur(self.image, (15, 15), 0)
        self.augmented_images.append({'name':'Blur', 'image':blurred_image})

    def adjust_contrast(self, contrast_factor):
        adjusted_image = cv2.convertScaleAbs(self.image, alpha=contrast_factor, beta=2)
        self.augmented_images.append({'name':'Contrast', 'image': adjusted_image})

    def brightness(self, brightness_factor):
        adjusted_image = cv2.convertScaleAbs(self.image, beta=brightness_factor)
        self.augmented_images.append({'name':'Brightness', 'image': adjusted_image})
    def projection(self, perspective_factor):
        rows, cols, _ = self.image.shape
        # Define source and destination points for perspective transformation
        src_points = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1], [cols - 1, rows - 1]])
        
        # Adjust destination points for a more pronounced perspective effect
        # dst_points = np.float32([[0, 0], [cols - 1, 0], [int(0.3 * cols), rows - 1], [int(0.7 * cols), rows - 1]])
        dst_points = np.float32([
            [int(0.1 * cols), int(0.3 * rows)],         # Top-left corner
            [int(0.9 * cols), int(0.1 * rows)],         # Top-right corner
            [int(0.3 * cols), int(0.9 * rows)],         # Bottom-left corner
            [int(0.9 * cols), int(0.6 * rows)]          # Bottom-right corner
        ])            # Calculate perspective transformation matrix
        perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

        # Apply perspective transformation
        perspective_transformed_image = cv2.warpPerspective(self.image, perspective_matrix, (cols, rows), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))


        self.augmented_images.append({'name':'Pojection', 'image':perspective_transformed_image})
    def flip_vertical(self):
        flipped_image = cv2.flip(self.image, 0)
        self.augmented_images.append({'name':'Flip', 'image': flipped_image})

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
            imagePath = f"images/{self.image_name}_{image['name']}.{self.image_extenssion}"
            print(imagePath)
            cv2.imwrite(imagePath, image['image'])
            
    def some_augmentations(self, num_operations):
        available_operations = [
            self.rotate,
            self.blur,
            self.adjust_contrast,
            # Add other augmentation methods as needed
        ]

        # Randomly shuffle the list of operations
        random.shuffle(available_operations)

        # Apply the selected number of operations
        for operation in available_operations[:max(num_operations - 1, 0)]:
            # Generate random parameters if needed for each operation
            if operation.__name__ == 'rotate':
                angle = random.uniform(-45, 45)
                operation(angle)
            elif operation.__name__ == 'blur':
                operation()
            elif operation.__name__ == 'adjust_contrast':
                factor = random.uniform(0.5, 2.0)
                operation(factor)
            # Add similar conditions for other operations

            
# Example usage
image_path = 'images/Apple/Apple_Black_rot/image (13).JPG'
augmentor = ImageAugmentor(image_path)

# Apply augmentations
augmentor.some_augmentations(3)
# augmentor.rotate(-25)
# augmentor.blur()
# augmentor.adjust_contrast(1.5)
# augmentor.scale(2)
# augmentor.brightness(60)
# augmentor.projection(0.8)
# augmentor.flip_vertical()
augmentor.save_images()
# augmentor.flip_vertical()

# Display original and augmented images in a horizontal plot
augmentor.show_augmented_images()
