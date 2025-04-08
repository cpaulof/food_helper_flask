import torch
from torchvision import transforms
from PIL import Image
import yaml
import os


with open("./cv/config.yaml", "r") as file:
    config= yaml.safe_load(file)
    file.close()

from cv.models import models as current_models


model_weights = {
    # 'custom_cnn': "./weights/custom_cnn_17M/best_model.pth",
    # 'custom_efficient_net': "./weights/custom_efficient_net_17M/best_model.pth",
    # 'custom_vit': "./weights/custom_vit_17M/best_model.pth",
    'pretrained_resnet50': "./cv/pretrained_resnet50_17M/best_model.pth",
    # 'unet': "",
}
device = config['device']
MODEL_NAME = "pretrained_resnet50"
MODEL_WEIGHT_PATH = model_weights[MODEL_NAME]

def load_model(model_name=MODEL_NAME, weight_dir=MODEL_WEIGHT_PATH):
    ModelClass = current_models[model_name]
    model = ModelClass()
    print("Modelo criado!!!")

    try:
        model.load_state_dict(torch.load(weight_dir, map_location=torch.device(device)))
    except Exception as e:
        print("Erro ao Carregar pesos do Modelo!")
        print(e)
        return
    print("pesos carregados!")
    return model

model = load_model()
model.eval()

mean_scaling = torch.tensor([254.93, 218.01, 12.71, 19.33, 18.10]).to(device)
std_scaling = torch.tensor([220.97, 163.06, 13.41, 22.29, 20.22]).to(device)


norm_image = lambda x: x*2-1.
transform_test = transforms.Compose([
    transforms.Resize((config['image_size'], config['image_size'])),
    transforms.ToTensor(),
    transforms.Lambda(norm_image)
])

def load_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img_input = transform_test(img)
    return img_input.unsqueeze(0)

@torch.no_grad
def inference(model, x):
    y = model(x)
    y = torch.exp(y)*mean_scaling
    y = y.numpy()
    print(y)
    return y

def preditct_from_flask_api(file):
    x = load_image(file)
    return inference(model, x)

if __name__ == "__main__":
    inference(model, load_image('test1.png'))
    inference(model, load_image('test2.png'))
