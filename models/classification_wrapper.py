import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import regnet_y_400mf
import re

class RegNetFeatureExtractor(nn.Module):
    def __init__(self):
        super(RegNetFeatureExtractor, self).__init__()
        self.regnet = regnet_y_400mf(pretrained=True)
        
        # Specify layers to extract features from
        self.feature_layers = ['stem', 'trunk_output', 'avgpool']
        
    def forward(self, x):
        features = []
        for name, layer in self.regnet.named_children():
            if name == 'fc':
                # Skip the fully connected layer
                continue
            x = layer(x)
            if name in self.feature_layers:
                features.append(x)

        # Flatten the output of the last layer ('avgpool')
        x = torch.flatten(x, 1)  # Flatten to (batch_size, num_features)
        
        return features, x   # return the selected features and the final output

class ModelLossWrapper:
    def __init__(self, model_path="../saved_models/regnet_y_400mf.pth", device='cuda'):
        self.model = self.load_model(model_path)
        self.model = self.model.to(device)
        self.model.eval()
        self.device = device

        # Initialize the RegNet feature extractor
        self.feature_extractor = RegNetFeatureExtractor().to(device)
        self.feature_extractor.eval()

        # Initialize weights for feature layers
        self.feature_weights = nn.Parameter(torch.ones(len(self.feature_extractor.feature_layers), device=self.device))

        # Parameters for Adaptive Learning Rate Adjustment
        self.alpha = 0.01  # Learning rate for the weight updates
        self.beta1 = 0.9   # Decay rate for the first moment
        self.beta2 = 0.999  # Decay rate for the second moment
        self.epsilon = 1e-8 # Small constant to prevent division by zero
        self.m_t = torch.zeros_like(self.feature_weights, device=self.device)
        self.v_t = torch.zeros_like(self.feature_weights, device=self.device)
        self.t = 0

    def load_model(self, model_path):
        """Load and return the entire model."""
        model = torch.load(model_path)
        return model
    
    def extract_label(self, image_path):
        # Extracts labels from filenames and categorizes into binary classes
        grade = image_path.split('_')[-1].replace('.png', '')
        return 1 if grade == '3+' else 0

    def compute_loss(self, input_images, img_paths, disc_predictions, usefor):
        """Compute the custom loss using the loaded model."""
        if usefor == "generator":
            labels = [self.extract_label(path) for path in img_paths]
            labels = torch.tensor(labels, dtype=torch.long, device=self.device)

            input_images = input_images.to(self.device)
            with torch.no_grad():
                outputs = self.model(input_images)
            
            loss = F.cross_entropy(outputs, labels)
            return loss
        
        if usefor == "discriminator":
            real_labels = torch.tensor([self.extract_label(path) for path in img_paths], dtype=torch.long, device=self.device)
            disc_predictions = disc_predictions.to(self.device)
            
            # Ensure disc_predictions is 2D with shape (batch_size, num_classes)
            if disc_predictions.dim() == 4:  # If discriminator output is in the form of patches
                disc_predictions = disc_predictions.mean(dim=[2, 3])  # Average the predictions over the spatial dimensions
            
            # Convert disc_predictions to binary labels
            _, predicted_classes = torch.max(disc_predictions, dim=1)
            predicted_binary = (predicted_classes == 3).long()  # Assuming class '3' is the '3+' class
            
            loss = F.cross_entropy(predicted_binary.unsqueeze(1).float(), real_labels.unsqueeze(1).float())
            return loss



    def compute_feature_matching_loss(self, real_images, fake_images, class_loss):
        # Extract features
        real_features, _ = self.feature_extractor(real_images)
        fake_features, _ = self.feature_extractor(fake_images)
        
        # Initialize loss
        loss = 0.0
        self.t += 1  # Time step for adaptive learning rate
        
        # Iterate over the selected feature maps
        for idx, (real_feat, fake_feat) in enumerate(zip(real_features, fake_features)):
            # Ensure both features have the same shape
            if real_feat.shape != fake_feat.shape:
                raise ValueError(f"Shape mismatch: {real_feat.shape} vs {fake_feat.shape}")
            
            # Compute L1 loss between real and fake features
            layer_loss = F.l1_loss(fake_feat, real_feat.detach())
            
            # Normalize the layer loss
            normalized_layer_loss = layer_loss / (torch.norm(layer_loss) + self.epsilon)
            
            # Calculate gradients for the adaptive learning rate
            g_t = normalized_layer_loss.detach()

            # Update biased first moment estimate
            self.m_t[idx] = self.beta1 * self.m_t[idx] + (1 - self.beta1) * g_t
            # Update biased second moment estimate
            self.v_t[idx] = self.beta2 * self.v_t[idx] + (1 - self.beta2) * (g_t ** 2)

            # Bias-corrected first and second moment estimates
            m_hat = self.m_t[idx] / (1 - self.beta1 ** self.t)
            v_hat = self.v_t[idx] / (1 - self.beta2 ** self.t)

            # Adaptive learning rate adjustment
            adaptive_lr = self.alpha / (torch.sqrt(v_hat) + self.epsilon)

            # Calculate the new weight without reassigning it to feature_weights
            new_weight = self.feature_weights[idx] + adaptive_lr * m_hat

            # classification loss adaptive weighting
            classification_weight = 1 / (class_loss + self.epsilon)

            # Apply the new weight to this layer's loss
            weighted_loss = new_weight * normalized_layer_loss * classification_weight
            
            loss += weighted_loss
        
        # Normalize the weights (optional, but can help stabilize training)
        with torch.no_grad():
            self.feature_weights /= self.feature_weights.sum()
        
        # Average the loss over the number of feature maps
        loss /= len(real_features)
        
        return loss



if __name__ == "__main__":
    # Check if CUDA is available, else use CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model_path = "saved_models/regnet_y_400mf.pth"
    wrapper = ModelLossWrapper(model_path, device=device)

    input_image = None
    img_paths = None
    disc_pred = None
    usefor = "generator"
    wrapper.compute_loss(input_image, img_paths, disc_pred ,usefor)