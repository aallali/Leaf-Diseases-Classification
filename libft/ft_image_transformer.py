from plantcv import plantcv as pcv
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import glob
import os


class Options:
    def __init__(self, src_path, dest_path="./tmp"):
        self.isDir = True  # assuming the src_path is a directory by default
        self.full_path = os.path.normpath(src_path)
        self.destination = os.path.normpath(dest_path)

        if os.path.isfile(src_path):
            self.image_name = os.path.basename(
                self.full_path
            ).replace(".JPG", "")
            directoryName = os.path.dirname(self.full_path)
            self.class_name = os.path.basename(directoryName)
            self.isDir = False
        elif os.path.isdir(src_path):
            self.directory = os.path.dirname(self.full_path)
        else:
            print("invalid source path")
            exit(1)

        return


class Transforner:
    def __init__(self, options):
        self.options = options

        # # apply write image
        # pcv.params.debug_outdir = self.options.outdir
        # if self.options.writeimg:
        #     self.name_save = self.options.outdir + "/" \
        #         + getlastname(self.options.image)

        # original
        self.img = None

        # gaussian_blur
        self.gauss = None

        # mask
        self.masked2 = None
        self.masked2_raw = None
        # ROI
        self.roi = None
        self.hierarchy3 = None
        self.kept_mask = None

        # analysis_image
        self.analysis_image = None

        # pseudolandmarks
        self.pseudolandmarks_image = None

    def getPath(self, suffix):
        dst = self.options.destination
        cName = self.options.class_name
        imgName = self.options.image_name
        Path(f"{dst}/{cName}").mkdir(parents=True, exist_ok=True)

        return f"{dst}/{cName}/{imgName}_{suffix}.JPG"

    def load_original(self, img_raw=None):
        if img_raw is not None:
            self.img = img_raw
        else:
            img, _, _ = pcv.readimage(self.options.full_path)
            self.img = img
        # Convert RGB to grayscale
        self.GRAY = pcv.rgb2gray(rgb_img=self.img)

        pcv.print_image(
            self.img,
            filename=self.getPath("original"),
        )
        return self.img

    def guassian_blur(self):
        if self.img is None:
            self.load_original()

        s = pcv.rgb2gray_hsv(rgb_img=self.img, channel="s")

        s_thresh = pcv.threshold.binary(
            gray_img=s, threshold=60, max_value=255, object_type="light"
        )
        self.gauss = pcv.gaussian_blur(
            img=s_thresh, ksize=(5, 5), sigma_x=0, sigma_y=None
        )
        pcv.print_image(
            self.gauss,
            filename=self.getPath("gaussian_blur"),
        )
        return self.gauss

    def mask(self):
        if self.gauss is None:
            self.guassian_blur()
        # Convert the 'b' channel of the LAB color space from the RGB image
        # to grayscale
        b = pcv.rgb2gray_lab(rgb_img=self.img, channel="b")

        # Create a binary threshold image from the 'b' channel
        b_thresh = pcv.threshold.binary(
            gray_img=b, threshold=137, max_value=255, object_type="light"
        )

        # Logical OR operation on two binary images
        bs = pcv.logical_or(bin_img1=self.gauss, bin_img2=b_thresh)

        # Apply the binary mask to the original image
        masked = pcv.apply_mask(img=self.img, mask=bs, mask_color="white")

        # Convert the 'a' and 'b' channels of the LAB color space from
        # the masked image to grayscale
        masked_a = pcv.rgb2gray_lab(rgb_img=masked, channel="a")
        masked_b = pcv.rgb2gray_lab(rgb_img=masked, channel="b")

        # Create binary threshold images for the 'a' channel of
        # the masked image
        maskeda_thresh = pcv.threshold.binary(
            gray_img=masked_a,
            threshold=115,
            max_value=255,
            object_type="dark"
        )
        maskeda_thresh1 = pcv.threshold.binary(
            gray_img=masked_a,
            threshold=135,
            max_value=255,
            object_type="light"
        )

        # Create a binary threshold image for the 'b' channel of
        # the masked image
        maskedb_thresh = pcv.threshold.binary(
            gray_img=masked_b,
            threshold=128,
            max_value=255,
            object_type="light"
        )

        # Logical OR operations on binary images to create composite masks
        ab1 = pcv.logical_or(bin_img1=maskeda_thresh, bin_img2=maskedb_thresh)
        ab = pcv.logical_or(bin_img1=maskeda_thresh1, bin_img2=ab1)

        # Fill holes in the 'ab' composite mask
        _AB = pcv.fill(bin_img=ab, size=200)

        # Apply the filled 'ab' mask to the original masked image
        _masked2 = pcv.apply_mask(img=masked, mask=_AB, mask_color="white")

        pcv.print_image(
                _masked2,
                filename=self.getPath("mask"),
        )

        self.masked2 = _masked2
        self.ab = _AB
        return Image.open(self.getPath("mask"))

    def roi_objects(self):
        if self.ab is None:
            self.mask()

        pcv.params.debug_outdir = self.options.destination
        pcv.params.debug = "print"

        id_objects, obj_hierarchy = pcv.find_objects(
            img=self.img,
            mask=self.ab
        )

        roi1, roi_hierarchy = pcv.roi.rectangle(
            img=self.img,
            x=0,
            y=0,
            h=250,
            w=250
        )

        roi, hierarchy3, kept_mask, _ = pcv.roi_objects(
            img=self.img,
            roi_contour=roi1,
            roi_hierarchy=roi_hierarchy,
            object_contour=id_objects,
            obj_hierarchy=obj_hierarchy,
            roi_type="partial",
        )

        file_rename = (
            self.options.destination
            + "/"
            + str(pcv.params.device - 2)
            + "_obj_on_img.png"
        )

        os.rename(file_rename, self.getPath("roi_objects"))

        for f in glob.glob(f"{self.options.destination}/*.png"):
            os.remove(f)

        pcv.params.debug = None

        self.roi = roi
        self.hierarchy3 = hierarchy3
        self.kept_mask = kept_mask

        return roi, hierarchy3, kept_mask

    def analysis_obj(self):
        if self.roi is None or self.hierarchy3 is None:
            self.roi_objects()

        pcv.params.debug = None

        obj, mask = pcv.object_composition(
            img=self.img, contours=self.roi, hierarchy=self.hierarchy3
        )

        analysis_image = pcv.analyze_object(
            img=self.img, obj=obj, mask=mask, label="default"
        )

        pcv.print_image(
                analysis_image,
                filename=self.getPath("analysis_obj")
        )

        self.mask1 = mask
        self.obj = obj
        self.analysis_image = analysis_image
        return analysis_image

    def pseudolandmarks(self):
        if self.mask1 is None or self.obj is None:
            self.pseudolandmarks()

        pcv.params.debug = "print"

        top_x, bottom_x, center_v_x = pcv.x_axis_pseudolandmarks(
            img=self.img, obj=self.obj, mask=self.mask1, label="default"
        )

        pcv.params.debug = None

        file_rename = (
            self.options.destination
            + "/"
            + str(pcv.params.device - 1)
            + "_x_axis_pseudolandmarks.png"
        )

        os.rename(file_rename, self.getPath("pseudolandmarks"))
        for f in glob.glob("./tmp/*.png"):
            os.remove(f)

        self.pseudolandmarks_image = Image.open(
            self.getPath("pseudolandmarks")
        )
        return self.pseudolandmarks_image

    def colors_histogram(self):
        pcv.params.debug = None
        color_histogram = pcv.analyze_color(
            rgb_img=self.img,
            mask=self.mask1,
            colorspaces="all",
            label="default",
        )
        print(color_histogram)
        pcv.print_image(
                color_histogram,
                filename=self.getPath("colors_histogram")
        )

        # colorsAnalayse = Image.open(self.getPath("colors_histogram"))
        # print(colorsAnalayse)
        # plt.figure()
        # plt.title("Colors Histogram")
        # plt.imshow(Image.open(self.getPath("colors_histogram")))
        # plt.axis('off')
        # plt.show(block=True)
        return color_histogram

    def run_all(self):
        # self.load_original()
        self.guassian_blur()
        self.mask()
        self.roi_objects()
        self.analysis_obj()
        self.pseudolandmarks()

    def load_all_transformations(self):
        images = {
            'Fig1. Original': Image.open(self.getPath("original")),
            'Fig2. Gaussian_Blur': Image.open(self.getPath("gaussian_blur")),
            'Fig3. Mask': Image.open(self.getPath("mask")),
            'Fig4. Roi_Objects': Image.open(self.getPath("roi_objects")),
            'Fig5. Pseudo-LandMarks': Image.open(
                self.getPath("pseudolandmarks")
            ),
            'Fig6. Analysis Obj.': Image.open(self.getPath("analysis_obj")),
        }
        return images

    def plot_all(self):
        images = self.load_all_transformations()

        # Variable for the number of images per row
        images_per_row = 3

        # Calculate the number of rows needed based on the number of
        # images and images per row
        num_rows = len(images) // images_per_row + len(images) % images_per_row

        # Create a multi-row figure with the specified number of images per row
        _, axes = plt.subplots(
            num_rows,
            images_per_row,
            figsize=(10, 5 * num_rows)
        )

        # Flatten the axes array for easier indexing
        axes = axes.flatten()

        # Iterate through the images and plot them in the figure
        for idx, (transformName, img_data) in enumerate(images.items()):
            axes[idx].imshow(img_data, cmap='gray')
            axes[idx].set_title(transformName)
            # axes[idx].axis('off')

        # Adjust layout for better spacing
        plt.tight_layout()
        plt.show(block=True)


def bulk_transformer(options):
    result = [
        os.path.join(dp, f) for dp, dn, filenames in
        os.walk(options.full_path) for f in filenames
        if os.path.splitext(f)[1] == '.JPG'
    ]

    for file in tqdm(result):
        file_options = Options(src_path=file, dest_path=options.destination)
        transformer = Transforner(file_options)
        transformer.run_all()
