import os
import cv2
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import jaccard_score
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

# Function to calculate Mutual Information
def calculate_mutual_information(image1, image2):
    hist_2d, _, _ = np.histogram2d(image1.ravel(), image2.ravel(), bins=20)
    pxy = hist_2d / float(np.sum(hist_2d))
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    px_py = px[:, None] * py[None, :]
    nzs = pxy > 0
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

# Function to calculate Normalized Cross-Correlation
def calculate_ncc(image1, image2):
    return pearsonr(image1.ravel(), image2.ravel())[0]

# Function to calculate Sum of Squared Differences (SSD)
def calculate_ssd(image1, image2):
    return np.sum((image1 - image2) ** 2)

# Function to calculate Structural Similarity Index (SSIM)
def calculate_ssim(image1, image2):
    return ssim(image1, image2, multichannel=True)

# Function to detect keypoints and compute matches
def calculate_keypoint_matching(image1, image2):
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image2, None)
    
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
    return len(good_matches), len(keypoints1), len(keypoints2)

# Function to calculate Jaccard Index (requires binary images or segmentation masks)
def calculate_jaccard_index(image1, image2):
    return jaccard_score(image1.ravel(), image2.ravel(), average='macro')

# Function to read and convert image to grayscale
def read_and_convert_to_grayscale(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at path {image_path} could not be loaded.")
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Main function to process image pairs and calculate metrics
def process_image_pairs(he_dir, ihc_dir, output_excel):
    print(f"Checking paths: H&E images directory: {he_dir}, IHC images directory: {ihc_dir}")
    
    if not os.path.exists(he_dir):
        raise FileNotFoundError(f"The H&E images directory does not exist: {he_dir}")
    if not os.path.exists(ihc_dir):
        raise FileNotFoundError(f"The IHC images directory does not exist: {ihc_dir}")

    he_files = sorted(os.listdir(he_dir))
    ihc_files = sorted(os.listdir(ihc_dir))

    if len(he_files) != len(ihc_files):
        raise ValueError("The number of H&E images and IHC images must be the same.")

    data = []

    for he_file, ihc_file in zip(he_files, ihc_files):
        if he_file != ihc_file:
            raise ValueError(f"Mismatch in filenames: {he_file} and {ihc_file}")

        he_path = os.path.join(he_dir, he_file)
        ihc_path = os.path.join(ihc_dir, ihc_file)

        he_image = read_and_convert_to_grayscale(he_path)
        ihc_image = read_and_convert_to_grayscale(ihc_path)

        # Calculate metrics
        mi = calculate_mutual_information(he_image, ihc_image)
        ncc = calculate_ncc(he_image, ihc_image)
        ssd = calculate_ssd(he_image, ihc_image)
        ssim_value = calculate_ssim(he_image, ihc_image)
        good_matches, keypoints1, keypoints2 = calculate_keypoint_matching(he_image, ihc_image)
        jaccard_index = calculate_jaccard_index(he_image > 127, ihc_image > 127)

        # Store results
        data.append({
            "Filename": he_file,
            "Mutual Information (Higher is Better)": mi,
            "Normalized Cross-Correlation (Higher is Better)": ncc,
            "Sum of Squared Differences (Lower is Better)": ssd,
            "Structural Similarity Index (Higher is Better)": ssim_value,
            "Keypoint Matches (Higher is Better)": good_matches,
            "Keypoints in Image 1": keypoints1,
            "Keypoints in Image 2": keypoints2,
            "Jaccard Index (Higher is Better)": jaccard_index
        })

    # Create DataFrame and save to Excel
    df = pd.DataFrame(data)
    df.to_excel(output_excel, index=False)
    print(f"Results saved to {output_excel}")


he_dir = 'datasets/BCI/A/train'  
ihc_dir = 'datasets/BCI/B/train'  
output_excel = 'data_preprocessing/train_image_pair_scores.xlsx'

process_image_pairs(he_dir, ihc_dir, output_excel)

he_dir = 'datasets/BCI/A/test'  
ihc_dir = 'datasets/BCI/B/test' 
output_excel = 'data_preprocessing/test_image_pair_scores.xlsx'

process_image_pairs(he_dir, ihc_dir, output_excel)
