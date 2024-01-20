# Todo List

- [x] Distribution
    - [x] scrap images recursively from a given path
    - [x] generate stats about total images under each sub folder
    - [x] calculate total augmenations to make in order to balance data
    - [x] plot a BAR + PIE chart
    - [x] encapsulate everything under one function `ft_distribution(target_path, totalVariants, plot_chart=False)`
    - [ ] make the function accesible via terminal run e.g: `python3 000-Distribution.py ./dataset/Apple`
- [ ] Augmenation
    - [x] make a an ImageAugmentor class to generate all possible 7 augmentations
    - [ ] encapsulate all in single function
    - [x] save the augmented images into same folder as original
    - [ ] use `ft_distribution` to augment all images inside given path
