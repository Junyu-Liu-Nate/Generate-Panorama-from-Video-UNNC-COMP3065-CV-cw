import os
import glob
import cv2
import numpy as np
# from basicsr.data.util import totensor
from basicsr.utils import img2tensor as _img2tensor, tensor2img, imwrite, scandir
from basicsr.utils.lmdb_util import make_lmdb_from_imgs
from pathlib import Path
from PIL import Image

# def generate_patches(sharp_dir, blur_dir, patch_size=256, stride=256):
#     """Generate patches from sharp and blur images."""
#     # Get the list of sharp and blur image paths
#     sharp_paths = sorted(glob.glob(os.path.join(sharp_dir, '*')))
#     blur_paths = sorted(glob.glob(os.path.join(blur_dir, '*')))
    
#     # Check that the number of sharp and blur images is the same
#     assert len(sharp_paths) == len(blur_paths), 'The number of sharp and blur images must be the same'
    
#     # Create lists to store the patches
#     sharp_patches = []
#     blur_patches = []
    
#     # Loop over the sharp and blur image paths
#     for sharp_path, blur_path in zip(sharp_paths, blur_paths):
#         # Load the sharp and blur images
#         sharp_img = cv2.imread(sharp_path)
#         blur_img = cv2.imread(blur_path)
        
#         # Check that the sharp and blur images have the same shape
#         assert sharp_img.shape == blur_img.shape, 'The shape of the sharp and blur images must be the same'
        
#         # Get the height and width of the images
#         h, w = sharp_img.shape[:2]
        
#         # Generate patches from the sharp and blur images
#         for i in range(0, h - patch_size + 1, stride):
#             for j in range(0, w - patch_size + 1, stride):
#                 # Extract a patch from the sharp image
#                 sharp_patch = sharp_img[i:i+patch_size, j:j+patch_size]
                
#                 # Extract a patch from the blur image
#                 blur_patch = blur_img[i:i+patch_size, j:j+patch_size]
                
#                 # Add the patches to the lists
#                 sharp_patches.append(sharp_patch)
#                 blur_patches.append(blur_patch)
    
#     return sharp_patches, blur_patches

# def totensor(img, bgr2rgb=False, float32=True):
#     img = img.astype(np.float32) / 255.
#     return _img2tensor(img, bgr2rgb=bgr2rgb, float32=float32)

# def prepare_data(sharp_dir, blur_dir, lmdb_path, patch_size=256, stride=256):
#     """Prepare training data for NAFNet."""
#     # Generate patches from the sharp and blur images
#     print('Generating patches...')
#     sharp_patches, blur_patches = generate_patches(sharp_dir, blur_dir, patch_size=patch_size, stride=stride)
    
#     # Convert the patches to tensors
#     print('Converting patches to tensors...')
#     data = []
#     for i in range(len(sharp_patches)):
#         # Convert the sharp patch to a tensor
#         sharp_tensor = totensor(sharp_patches[i])
        
#         # Convert the blur patch to a tensor
#         blur_tensor = totensor(blur_patches[i])
        
#         # Add a new dimension to the tensors (for LMDB)
#         data.append((sharp_tensor.unsqueeze(0), blur_tensor.unsqueeze(0)))
    
#     # Create an LMDB file from the data
#     print('Creating LMDB file...')
#     make_lmdb_from_imgs(lmdb_path, data)

def create_lmdb_dataset(blur_folder, sharp_folder, lmdb_path):
    blur_folder = Path(blur_folder)
    sharp_folder = Path(sharp_folder)
    
    blur_images = sorted(blur_folder.glob('*.png'))
    sharp_images = sorted(sharp_folder.glob('*.png'))
    
    assert len(blur_images) == len(sharp_images), "Number of images in both folders should be the same"
    
    img_path_list = []
    keys = []
    for i, (blur_img_path, sharp_img_path) in enumerate(zip(blur_images, sharp_images)):
        img_path_list.extend([str(blur_img_path.resolve()), str(sharp_img_path.resolve())])
        keys.extend([f"{i:08d}_blur", f"{i:08d}_sharp"])
    
    make_lmdb_from_imgs(str(blur_folder), lmdb_path, img_path_list, keys)

# Set the paths to your sharp and blur image directories
# sharp_dir = '../../dataset/sharp_images/'
# blur_dir = '../../dataset/blurred_images/'
sharp_dir = 'datasets/sharp_frames/'
blur_dir = 'datasets/blurred_frames/'

# Set the path to save the LMDB file
lmdb_path = 'datasets/customized_data.lmdb'

# Set the patch size and stride (optional)
patch_size = 256
stride = 256

# Prepare your training data for NAFNet
create_lmdb_dataset(sharp_dir, blur_dir, lmdb_path)