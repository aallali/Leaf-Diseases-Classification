# Todo List

- [x] Distribution
    - [x] scrap images recursively from a given path
    - [x] generate stats about total images under each sub folder
    - [x] calculate total augmenations to make in order to balance data
    - [x] plot a BAR + PIE chart
    - [x] encapsulate everything under one function `ft_distribution(target_path, totalVariants, plot_chart=False)`
    - [x] make the function accesible via terminal run e.g: `python3 000-Distribution.py ./dataset/Apple`
    - [x] split data between `validation`/`train` datasets
        - [x] specify flag for it
- [x] Augmenation
    - [x] make a an ImageAugmentor class to generate all possible 7 augmentations
    - [x] encapsulate all in single function
    - [x] save the augmented images into same folder as original
    - [x] use `ft_distribution` to augment all images inside given path
    - [x] save augmented images in `augmented_directory` by default or from user by args
    - [x] plot augmented images if a single image is given as argument


- [x] Transformation
    - [x] make the transformation functions:
        - [x] guassian_blur
        - [x] mask
        - [x] roi (range of interests) objects
        - [x] analysis object
        - [x] pseudo-landmarks
        - [x] colors historgram
    - [x] if given path is image transform it and show output in place
    - [x] if given path is directory then transform contained images recursively with their class name to destination project