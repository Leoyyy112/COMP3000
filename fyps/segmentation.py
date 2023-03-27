# # segmentation.py

import numpy as np
from PIL import Image
import imageio
import os
import glob
import skimage.io as io
import skimage.transform as trans
import os
import os.path
import tensorflow
import albumentations
import cv2
import matplotlib.pyplot as plt
from segmentation_models import Unet
import segmentation_models as sm
from tensorflow.keras.optimizers import Adam
SOURCE_SIZE = 512
TARGET_SIZE = 256

val_augs = albumentations.Compose([
    albumentations.Resize(TARGET_SIZE, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
])


def png_to_npy(png_file_path):
    # Read the PNG file using Pillow
    img = Image.open(png_file_path)

    # Convert the image to a NumPy array
    img = np.array(img)

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


def fscore_glass(y_true, y_pred):
    return sm.metrics.f1_score(y_true[..., 0:1],
                               y_pred[..., 0:1])


def fscore_consolidation(y_true, y_pred):
    return sm.metrics.f1_score(y_true[..., 1:2],
                               y_pred[..., 1:2])


def fscore_lungs_other(y_true, y_pred):
    return sm.metrics.f1_score(y_true[..., 2:3],
                               y_pred[..., 2:3])


def fscore_glass_and_consolidation(y_true, y_pred):
    return sm.metrics.f1_score(y_true[..., :2],
                               y_pred[..., :2])


def fscore_mean(y_true, y_pred):
    return 2 - fscore_glass_and_consolidation(y_true, y_pred) - fscore_lungs_other(y_true, y_pred)
def classify_masks(mask_batch, threshold=0.5):
    glass_flag = False
    consolidations_flag = False
    healthy_flag = False
    if np.any(mask_batch[0, :, :, 0] > threshold):
        glass_flag = True
    if np.any(mask_batch[0, :, :, 1] > threshold):
        consolidations_flag = True
    if not (glass_flag or consolidations_flag):
        healthy_flag = True
    return glass_flag, consolidations_flag, healthy_flag
def run_segmentation(image_path):
    # Load the saved model
    model = tensorflow.keras.models.load_model('models/Covidunet.h5',compile=False,custom_objects={
                                                'fscore_mean': fscore_mean,
                                                'fscore_consolidation': fscore_consolidation,
                                                'fscore_glass': fscore_glass,
                                                'fscore_lungs_other': fscore_lungs_other,
                                                'fscore_glass_and_consolidation': fscore_glass_and_consolidation})
    model.compile(Adam(learning_rate=0.001, amsgrad=True),
                  loss=fscore_mean)
    # Load the image and preprocess it for segmentation
    test_images_medseg = png_to_npy(image_path)
    image_batch = np.stack([val_augs(image=img)['image'] for img in test_images_medseg], axis=0)
    test_preds = model.predict_on_batch(image_batch)
    test_masks_prediction = test_preds > 0.5
    mask_image = visualize(test_masks_prediction)
    mask_image = (mask_image * 255).astype(np.uint8)
    mask_image = Image.fromarray(mask_image)
    mask_image.save('media/mask.png')
    return mask_image
def run_classification(image_path):
    model = tensorflow.keras.models.load_model('models/Covidunet.h5', compile=False, custom_objects={
        'fscore_mean': fscore_mean,
        'fscore_consolidation': fscore_consolidation,
        'fscore_glass': fscore_glass,
        'fscore_lungs_other': fscore_lungs_other,
        'fscore_glass_and_consolidation': fscore_glass_and_consolidation})
    model.compile(Adam(learning_rate=0.001, amsgrad=True),
                  loss=fscore_mean)

    test_images_medseg = png_to_npy(image_path)
    image_batch = np.stack([val_augs(image=img)['image'] for img in test_images_medseg], axis=0)
    test_preds = model.predict_on_batch(image_batch)
    test_masks_prediction = test_preds > 0.5
    flag1,flag2,flag3 = classify_masks(test_masks_prediction)
    return flag1,flag2,flag3
