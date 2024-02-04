#!/usr/bin/env python3

import os
import matplotlib.pyplot as plt
import argparse
from libft import (
    ft_scrap_images,
    ft_remove_prefix,
    ft_generate_random_hexa_color_codes,
    find_end_level_dicts,
    ft_split_dataset
)

COLORS = ft_generate_random_hexa_color_codes(100)
COLORS = [
            '#65735d', '#9e1aba', '#3450ea', '#e1833b', '#aff734', '#d992a6',
            '#454443', '#184398', '#ea82c1', '#ce5d61', '#514c16', '#0d7c38',
            '#7b0cdf', '#b7a250', '#2ed964', '#8f6b27', '#b40ded', '#feab37',
            '#dfa940', '#4f26ec', '#c39b46', '#323200', '#c00892', '#534c02',
            '#72815f', '#090b46', '#3bfd3e', '#60109b', '#0c9c13', '#c214c6',
            '#f72332', '#421b89', '#57aa48', '#ce5d65', '#95b520'
        ]


def build_image_stats(paths):
    stats = {}

    for path in paths:
        path = path.replace("\\", "/")
        path = ft_remove_prefix(path, "../")
        path = ft_remove_prefix(path, "./")
        path = ft_remove_prefix(path, "/")
        components = path.split("/")
        current_dict = stats

        for component in components[:-2]:
            current_dict = current_dict.setdefault(component, {})

        category = components[-2]
        current_dict[category] = current_dict.get(category, 0) + 1

    if len(stats) == 1:
        firstKey = next(iter(stats))
        if len(stats[firstKey]) == 1:
            stats = stats[firstKey]

    return stats


def plot_image_stats(title, stats):
    flattened_stats = find_end_level_dicts(stats)

    categories = list([d[0] for d in flattened_stats])
    counts = list([d[1] for d in flattened_stats])

    plt.figure(figsize=(12, 6))

    # hide_parent_folder
    plt.suptitle(f"Images distribution in {title}")

    # Pie chart
    plt.subplot(1, 2, 1)

    plt.pie(counts, labels=categories, autopct="%1.1f%%", colors=COLORS)
    plt.title("Pie chart")

    # Bar chart
    plt.subplot(1, 2, 2)

    bars = plt.bar(categories, counts, color=COLORS, width=0.6)
    plt.xticks(rotation=30, ha="right")

    plt.grid(True)
    plt.title("Bar chart")

    # Add legends
    plt.legend(
        bars,
        [f"{c.ljust(25)}{counts[i]}" for (i, c) in enumerate(categories)],
        title="Categories",
        bbox_to_anchor=(1.05, 0),
        loc="lower left",
    )

    plt.tight_layout()
    plt.savefig(f"./{title}_distribution", bbox_inches="tight")
    plt.show()


def ft_distribution(target_path, totalVariants, plot_chart=False):
    all_images = ft_scrap_images(target_path)
    folders_stats = build_image_stats(all_images)
    parent_folder_name = next(iter(folders_stats))
    end_level_folders_stats = find_end_level_dicts(folders_stats)

    lowest = 999999
    result = {
        "folder_statistics": {},
        "total_augmentation_to_balance": 0,
        "root_path": target_path,
        "image_paths": {},
    }

    for subFolder in end_level_folders_stats:
        folder_name = subFolder[0]
        total_imgs = subFolder[1]
        if total_imgs < lowest:
            lowest = total_imgs

        images_related_to_subfolder = [
            img for img in all_images if folder_name in img
        ]
        head, _ = os.path.split(images_related_to_subfolder[0])

        subFolder_images = [
            os.path.split(img)[1] for img in images_related_to_subfolder
        ]

        result["image_paths"][folder_name] = {
            "images": subFolder_images,
            "path_to_folder": head,
        }
    totalToAugment = totalVariants * lowest

    result["folder_statistics"] = end_level_folders_stats
    result["total_augmentation_to_balance"] = totalToAugment

    if plot_chart:
        plot_image_stats(
            title=parent_folder_name,
            stats=folders_stats
        )

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Process images in a specified folder."
    )
    parser.add_argument(
        "folder_path",
        help="Path to the folder containing images."
    )

    parser.add_argument(
        "-s", "--split",
        type=float,
        help="Specify the split ratio (validation data split) as a float \
            (default: 0.1, valid values : 0-1).",
        nargs='?', const=0.1,
    )
    args = parser.parse_args()
    [folder_path, split] = [args.folder_path, args.split]

    if split:
        if split > 0 and split < 1:
            train_dir_name = os.path.normpath(folder_path) + "_train_split"
            test_dir_name = os.path.normpath(folder_path) + "_test_split"

            print("Goal:")
            print(f"- {train_dir_name}      : {1 - split}")
            print(f"- {test_dir_name} : {split}")
            print("Splitting...")

            # Example usage:
            ft_split_dataset(
                folder_path,
                train_dir_name,
                test_dir_name,
                split
            )
        else:
            print("Invalid split value, please specify a value between 0-1")
            exit(1)
    else:

        distribution = ft_distribution(
            folder_path,
            totalVariants=6,
            plot_chart=True
        )

        # del distribution["image_paths"]
        for stat in distribution["folder_statistics"]:
            print(stat)


if __name__ == "__main__":
    main()
