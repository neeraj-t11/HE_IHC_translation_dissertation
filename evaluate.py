import os
import cv2 as cv
from PIL import Image
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from tqdm import tqdm
import argparse
import torch
import torchvision.transforms as transforms
from torchvision.models import inception_v3
import lpips
from scipy.linalg import sqrtm
import numpy as np
from torch.nn.functional import adaptive_avg_pool2d

def parse_opt():
    parser = argparse.ArgumentParser(description='Evaluate options')
    parser.add_argument('--result_path', type=str, default='./results/pyramidpix2pix', help='results saved path')
    opt = parser.parse_args()
    return opt

opt = parse_opt()

# Load LPIPS model
lpips_model = lpips.LPIPS(net='alex').cuda()

# Load pre-trained VGG16 model for perceptual loss
vgg = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True).features[:16].eval().cuda()
for param in vgg.parameters():
    param.requires_grad = False

# Load InceptionV3 model for FID and Inception Score
inception = inception_v3(pretrained=True, transform_input=False).eval().cuda()
for param in inception.parameters():
    param.requires_grad = False

transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def perceptual_loss(fake, real):
    fake_vgg = vgg(fake)
    real_vgg = vgg(real)
    return torch.nn.functional.mse_loss(fake_vgg, real_vgg)

def calculate_fid(fake_acts, real_acts):
    # Calculate mean and covariance for real and fake activations
    mu_fake = np.mean(fake_acts, axis=0)
    mu_real = np.mean(real_acts, axis=0)
    sigma_fake = np.cov(fake_acts, rowvar=False)
    sigma_real = np.cov(real_acts, rowvar=False)
    
    # Calculate FID
    diff = mu_fake - mu_real
    covmean = sqrtm(sigma_fake.dot(sigma_real))
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma_fake + sigma_real - 2 * covmean)
    return fid

def inception_score(preds, splits=10):
    scores = []
    preds = np.exp(preds)  # undo the log in softmax
    for i in range(splits):
        part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
        py = np.mean(part, axis=0)
        scores.append(np.exp(np.mean(np.sum(part * np.log(part / py), axis=1))))
    return np.mean(scores), np.std(scores)

def calculate_fid_and_inception(real_features, fake_features):
    # FID score
    fid_score = calculate_fid(fake_features['activations'], real_features['activations'])
    
    # Inception Score
    inception_mean, inception_std = inception_score(fake_features['preds'])

    return fid_score, inception_mean, inception_std

def extract_inception_features(images):
    preds = []
    activations = []

    with torch.no_grad():
        for img in images:
            img = img.cuda()
            
            # Pass through the Inception model
            x = inception.Conv2d_1a_3x3(img)
            x = inception.Conv2d_2a_3x3(x)
            x = inception.Conv2d_2b_3x3(x)
            x = inception.maxpool1(x)
            x = inception.Conv2d_3b_1x1(x)
            x = inception.Conv2d_4a_3x3(x)
            x = inception.maxpool2(x)
            x = inception.Mixed_5b(x)
            x = inception.Mixed_5c(x)
            x = inception.Mixed_5d(x)
            x = inception.Mixed_6a(x)
            x = inception.Mixed_6b(x)
            x = inception.Mixed_6c(x)
            x = inception.Mixed_6d(x)
            x = inception.Mixed_6e(x)
            x = inception.Mixed_7a(x)
            x = inception.Mixed_7b(x)
            x = inception.Mixed_7c(x)

            # Pool the output to match the expected input size for fc layer
            pooled_output = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))

            # Flatten the pooled output
            flattened_output = pooled_output.view(pooled_output.size(0), -1)

            # Pass through the fully connected layer
            preds.append(inception.fc(flattened_output).cpu().numpy())

            activations.append(flattened_output.cpu().numpy().flatten())

    preds = np.concatenate(preds, axis=0)
    activations = np.concatenate(activations, axis=0)
    return {'preds': preds, 'activations': activations}



def psnr_ssim_fid_and_is(result_path):
    psnr = []
    ssim = []
    perceptual_losses = []
    lpips_scores = []

    real_images = []
    fake_images = []

    for i in tqdm(os.listdir(os.path.join(result_path, 'test_latest/images'))):
        if 'fake_B' in i:
            try:
                fake_path = os.path.join(result_path, 'test_latest/images', i)
                real_path = os.path.join(result_path, 'test_latest/images', i.replace('fake_B', 'real_B'))
                
                print(f"Processing fake image: {fake_path}")
                print(f"Processing real image: {real_path}")
                
                fake = cv.imread(fake_path)
                real = cv.imread(real_path)

                if fake is None:
                    raise ValueError(f"Failed to read fake image from {fake_path}")
                if real is None:
                    raise ValueError(f"Failed to read real image from {real_path}")

                # Convert images to RGB for perceptual and LPIPS
                fake_rgb = cv.cvtColor(fake, cv.COLOR_BGR2RGB)
                real_rgb = cv.cvtColor(real, cv.COLOR_BGR2RGB)

                fake_tensor = transform(Image.fromarray(fake_rgb)).unsqueeze(0).cuda()
                real_tensor = transform(Image.fromarray(real_rgb)).unsqueeze(0).cuda()

                # PSNR and SSIM
                PSNR = peak_signal_noise_ratio(fake, real)
                psnr.append(PSNR)

                SSIM = structural_similarity(fake, real, multichannel=True, channel_axis=-1)
                ssim.append(SSIM)

                # Perceptual Loss
                perceptual_loss_value = perceptual_loss(fake_tensor, real_tensor).item()
                perceptual_losses.append(perceptual_loss_value)

                # LPIPS
                lpips_score = lpips_model(fake_tensor, real_tensor).item()
                lpips_scores.append(lpips_score)

                # Collect images for FID and Inception Score
                fake_images.append(fake_tensor)
                real_images.append(real_tensor)

            except Exception as e:
                print(f"Error processing {i}: {e}")
        else:
            continue
    
    if psnr and ssim and perceptual_losses and lpips_scores:
        average_psnr = sum(psnr) / len(psnr)
        average_ssim = sum(ssim) / len(ssim)
        average_perceptual_loss = sum(perceptual_losses) / len(perceptual_losses)
        average_lpips_score = sum(lpips_scores) / len(lpips_scores)
        
        print(f"The average PSNR (Peak Signal-to-Noise Ratio) is {average_psnr}")
        print(f"The average SSIM (Structural Similarity Index) is {average_ssim}")
        print(f"The average Perceptual Loss is {average_perceptual_loss}")
        print(f"The average LPIPS (Learned Perceptual Image Patch Similarity) score is {average_lpips_score}")

        # # Calculate FID and Inception Score
        # real_features = extract_inception_features(real_images)
        # fake_features = extract_inception_features(fake_images)

        # fid_score, inception_mean, inception_std = calculate_fid_and_inception(real_features, fake_features)
        
        # print(f"The FID (Fréchet Inception Distance) score is {fid_score}")
        # print(f"The Inception Score is {inception_mean} ± {inception_std}")
    else:
        print("No valid images found for PSNR, SSIM, Perceptual Loss, LPIPS, FID, and Inception Score calculation")

psnr_ssim_fid_and_is(opt.result_path)


# results for ./results/pix2pix_resnet_9blocks_PatchGAN_noclassifier_db
# The average PSNR is 19.273616199886373
# The average SSIM is 0.33018498258557233

# results for ./results/pix2pix_resnet_9blocks_PatchGAN_classifier_db
# The average PSNR is 19.755811626900478
# The average SSIM is 0.32785041627076295

# results for ./results/pix2pix_resnet_9blocks_PatchGAN_classifier25_db
# The average PSNR (Peak Signal-to-Noise Ratio) is 19.646019012619462
# The average SSIM (Structural Similarity Index) is 0.3676450059974264
# The average Perceptual Loss is 2.7060059459084984
# The average LPIPS (Learned Perceptual Image Patch Similarity) score is 0.3973351801643596