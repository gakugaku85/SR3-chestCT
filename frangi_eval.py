import os
import numpy as np
from skimage import io
from skimage.filters import frangi
from skimage.metrics import mean_squared_error
import SimpleITK as sitk
from icecream import ic
from pathlib import Path

def apply_frangi(image):
    """
    Apply Frangi filter to the inverted input image.
    :param image: numpy array, input image.
    :return: numpy array, Frangi filter output.
    """
    return frangi(255 - image)

def scale_to_uint8(image):
    """
    Scale the input image to 0-255 and convert to uint8.
    :param image: numpy array, input image.
    :return: numpy array, scaled image in uint8 format.
    """
    scaled = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
    return scaled.astype(np.uint8)

def crop_center(image, crop_size):
    """
    Crop the center of the image to the specified size.
    :param image: numpy array, input image.
    :param crop_size: tuple of two ints, (height, width) of the crop.
    :return: numpy array, cropped image.
    """
    h, w = image.shape
    ch, cw = crop_size
    start_h = (h - ch) // 2
    start_w = (w - cw) // 2
    return image[start_h:start_h + ch, start_w:start_w + cw]

def save_image(image, output_path):
    """
    Save the image as a PNG file.
    :param image: numpy array, input image.
    :param output_path: str, path to save the image.
    """
    io.imsave(output_path, image)

def evaluate_images(dir, num_images=5, ite=12000):
    """
    Evaluate the super-resolution images against high-resolution images.
    :param dir: str, directory containing HR and SR images.
    :param num_images: int, number of images to evaluate.
    :param ite: int, iteration number for the image filenames.
    :return: float, average MSE of the evaluated images.
    """
    crop_size = (255, 255)
    mse_values = []
    output_path = Path(dir) / "frangi"
    output_path.mkdir(parents=True, exist_ok=True)

    for i in range(1, num_images + 1):
        # Construct filenames
        hr_filename = f"{ite}_{i}_hr.mhd"
        sr_filename = f"{ite}_{i}_sr.mhd"
        inf_filename = f"{ite}_{i}_inf.mhd"

        # Load images
        hr_image = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(dir, hr_filename)))
        sr_image = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(dir, sr_filename)))
        inf_image = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(dir, inf_filename)))

        # ic(hr_image.max(), hr_image.min())
        # ic(sr_image.max(), sr_image.min())

        # Apply Frangi filter
        hr_frangi = apply_frangi(hr_image)*255
        sr_frangi = apply_frangi(sr_image)*255
        inf_frangi = apply_frangi(inf_image)*255

        # ic(hr_frangi.max(), hr_frangi.min())
        # ic(sr_frangi.max(), sr_frangi.min())

        # Crop the center
        hr_cropped = crop_center(hr_image, crop_size)
        sr_cropped = crop_center(sr_image, crop_size)
        inf_cropped = crop_center(inf_image, crop_size)
        hr_f_cropped = crop_center(hr_frangi, crop_size)
        sr_f_cropped = crop_center(sr_frangi, crop_size)

        # Scale images to 0-255 for saving
        hr_cropped_scaled = scale_to_uint8(hr_cropped)
        sr_cropped_scaled = scale_to_uint8(sr_cropped)
        inf_cropped_scaled = scale_to_uint8(inf_cropped)
        hr_f_cropped_scaled = scale_to_uint8(hr_f_cropped)
        sr_f_cropped_scaled = scale_to_uint8(sr_f_cropped)

        # Save cropped images
        save_image(hr_cropped_scaled, output_path / f"{ite}_{i}_hr_cropped.png")
        save_image(sr_cropped_scaled, output_path / f"{ite}_{i}_sr_cropped.png")
        save_image(inf_cropped_scaled, output_path / f"{ite}_{i}_inf_cropped.png")
        save_image(hr_f_cropped_scaled, output_path / f"{ite}_{i}_hr_f_cropped.png")
        save_image(sr_f_cropped_scaled, output_path / f"{ite}_{i}_sr_f_cropped.png")

        # Compute MSE
        mse = mean_squared_error(hr_frangi, sr_frangi)

        ic(mse)

        mse_values.append(mse)

    # Compute average MSE
    average_mse = np.mean(mse_values)
    return average_mse

# Usage example:
# dir = "experiments/sr_patch_64_val_241129_083024/results" # wdのtest
# dir = "experiments/sr_patch_64_val_241129_082326/results" # oriのtest

dir = "experiments/sr_patch_64_241125_090218/results/2400"
epoch = int(dir.split("/")[-1])
ite = round(epoch * 30, -2)

print(f"ite: {ite}")
num_images = 10 if dir.split("/")[-1] == "results" else 5

average_mse = evaluate_images(dir, num_images=num_images, ite=ite)
print(f"Average MSE across {num_images} images: {average_mse}")
