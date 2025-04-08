import torch
from torchvision import transforms, models
import yaml 

with open("./cv/config.yaml", "r") as file:
    config= yaml.safe_load(file)
    file.close()


class Resnet50(torch.nn.Module):
    def __init__(self, num_classes=config['num_classes']):
        super(Resnet50, self).__init__()
        resnet_config = config['models']['resnet50_pretrained']
        self.model = models.resnet50(num_classes=resnet_config['classes'])
        # self.model.load_state_dict(torch.load(resnet_config['weights'], map_location=config['device']))
        for param in self.model.parameters():
            param.requires_grad = False
        in_features = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(in_features=in_features, out_features=num_classes)
    
    def forward(self, x):
        return self.model(x)



if __name__ == "__main__":
    model = Resnet50()
    inputs = torch.normal(0, 1, (1, 3, config['image_size'], config['image_size']))
    outputs = model(inputs)
    print(outputs.shape)
    print("parameters:", sum(p.numel() for p in model.parameters() if not p.requires_grad))


