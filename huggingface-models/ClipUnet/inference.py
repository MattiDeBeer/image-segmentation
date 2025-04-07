import torch
from load_model import load_model
from torchvision import transforms
from PIL import Image

model = load_model()

preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

def predict(image: Image.Image):
    input_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
    return output.argmax(dim=1).squeeze().cpu().numpy()
