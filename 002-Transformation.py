#!/usr/bin/env python3

import argparse
from libft import Options, Transforner, bulk_transformer
import os


def main():
    parser = argparse.ArgumentParser(
        description="Extract features from leaf image."
    )
    parser.add_argument(
        "src_path",
        help="Path to the image to transform or folder containing images."
    )

    parser.add_argument(
        "-dst", "--destination",
        type=str,
        default="./tmp",
        help="the destination folder to save the transformed images",
    )

    args = parser.parse_args()

    if args.destination == "./tmp":
        args.destination = os.path.normpath(args.src_path) + "_transformed"

    options = Options(src_path=args.src_path, dest_path=args.destination)

    if not options.isDir:
        options = Options(src_path=args.src_path, dest_path="./tmp")
        print("Image name : ", options.image_name)
    else:
        print("Source directory : ", options.full_path)
    print("Destination directory : ", options.destination)

    if options.isDir:
        # bulk transformer
        print("Bulk transformer is running now, please be patient...")
        bulk_transformer(options)
        exit(1)
    else:
        # single transform
        transformer = Transforner(options)

        transformer.run_all()
        transformer.plot_all()

        transformer.colors_histogram()
        exit(1)


if __name__ == "__main__":
    main()
