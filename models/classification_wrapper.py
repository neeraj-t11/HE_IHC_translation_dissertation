import torch
import torch.nn.functional as F

class ModelLossWrapper:
    def __init__(self, model_path="saved_models/resnet100", device='cuda'):
        self.model = self.load_model(model_path)
        self.model = self.model.to(device)
        self.model.eval()
        self.device = device

    def load_model(self, model_path):
        """Load and return the entire model."""
        model = torch.load(model_path)
        return model

    def compute_loss(self, input_image):
        """Compute the custom loss using the loaded model."""
        input_image = input_image.to(self.device)
        with torch.no_grad():
            output = self.model(input_image)
        # Example loss: Mean squared error of the outputs (modify according to actual use-case)
        loss = F.mse_loss(output, torch.zeros_like(output))
        return loss

    def integrate_loss(self, gan_model, input_image, *args, **kwargs):
        """
        Integrate the computed loss into the GAN model's training routine.
        `gan_model` should be an instance of one of the GAN models that has a method `train_step`
        that can accept an additional loss parameter.
        """
        additional_loss = self.compute_loss(input_image)
        # Here, we assume 'train_step' can handle additional loss
        return gan_model.train_step(input_image, additional_loss, *args, **kwargs)



if __name__ == "__main__":
    # Check if CUDA is available, else use CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model_path = "saved_models/resnet100.pth"
    wrapper = ModelLossWrapper(model_path, device=device)


# Simulate a training loop
# gan_model = SomeGANModel()
# for input_image in dataloader:
#    loss = wrapper.integrate_loss(gan_model, input_image)
#    loss.backward()
#    optimizer.step()
