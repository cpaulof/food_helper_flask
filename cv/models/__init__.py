# from .custom_cnn import CustomCNN
# from .custom_efficientnet import CustomModelEfficientNet
# from .custom_vit import CustomViT
from .pretrained_resnet50 import Resnet50
# from .unet_vgg import CombinedModel

models = {
    # 'custom_cnn': CustomCNN,
    # 'custom_efficient_net': CustomModelEfficientNet,
    # 'custom_vit': CustomViT,
    'pretrained_resnet50': Resnet50,
    # 'unet': CombinedModel,
}