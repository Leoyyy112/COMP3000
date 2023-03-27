import numpy as np
from PIL import Image
import imageio
import os
import glob
import skimage.io as io
import skimage.transform as trans
import os
import os.path
from keras.models import load_model
import albumentations
import cv2
import matplotlib.pyplot as plt

SOURCE_SIZE = 512
TARGET_SIZE = 256

val_augs = albumentations.Compose([
    albumentations.Resize(TARGET_SIZE, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
])


def png_to_npy(png_file_path):
    # Read the PNG file
    img = imageio.imread(png_file_path)
    # Make sure the image has the expected dimensions
    if img.shape != (512, 512):
        raise ValueError(f"Image dimensions {img.shape} do not match the expected (512, 512) size.")
    # Convert the uint16 pixel values to the range of 0 to 1 (float64)
    img = img.astype(np.float64) / 65535
    # Rescale the pixel values from the range of 0 to 1 to the range of -3 to 3
    img = (img * (3 - (-3))) + (-3)
    # Reshape the image to (1, 512, 512, 1)
    img = img.reshape((1, 512, 512, 1))
    return img

def visualize(mask_batch):
    num_classes = mask_batch.shape[-1]
    mask_to_show = np.zeros((*mask_batch.shape[1:3], 3))

    for j in range(num_classes):
        mask_to_show[mask_batch[0, :, :, j] > 0.5] = plt.cm.get_cmap('viridis', num_classes)(j)[:3]

    return mask_to_show



def run_segmentation(image_path):
    # Load the saved model
    model = load_model('models/Covidunet.h5')

    # Load the image and preprocess it for segmentation
    test_images_medseg = png_to_npy(image_path)
    image_batch = np.stack([val_augs(image=img)['image'] for img in test_images_medseg], axis=0)
    test_preds = model.predict_on_batch(image_batch)
    test_masks_prediction = test_preds > 0.5
    mask_image = visualize(test_masks_prediction)

    mask_image = Image.fromarray(mask_image)
    mask_image.save('media/mask.png')
    return mask_image