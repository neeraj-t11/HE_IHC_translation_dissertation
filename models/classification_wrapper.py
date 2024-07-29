import torch
import torch.nn.functional as F
import re

class ModelLossWrapper:
    def __init__(self, model_path="../saved_models/resnet100", device='cuda'):
        self.model = self.load_model(model_path)
        self.model = self.model.to(device)
        self.model.eval()
        self.device = device

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
            real_labels = torch.tensor([self.extract_label(path) for path in img_paths], device=self.device)
            disc_predictions = disc_predictions.to(self.device)
            
            loss = F.cross_entropy(disc_predictions, real_labels)
            return loss

if __name__ == "__main__":
    # Check if CUDA is available, else use CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model_path = "saved_models/resnet100.pth"
    wrapper = ModelLossWrapper(model_path, device=device)

    input_image = None
    img_paths = None
    disc_pred = None
    usefor = "generator"
    wrapper.compute_loss(input_image, img_paths, disc_pred ,usefor)

# Simulate a training loop
# gan_model = SomeGANModel()
# for input_image in dataloader:
#    loss = wrapper.integrate_loss(gan_model, input_image)
#    loss.backward()
#    optimizer.step()
