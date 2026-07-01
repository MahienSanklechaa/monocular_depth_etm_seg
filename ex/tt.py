# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# print(torch.cuda.is_available())  
# from models.imagenet import mobilenetv2
# model = mobilenetv2()

# # Create a dummy input (batch_size=1, 3 channels, 224x224 resolution)
# dummy_input = torch.randn(1, 3,1056,320)

# # Forward pass through the model
# with torch.no_grad():
#     output = model(dummy_input)
    
# print(f"Input shape: {dummy_input.shape}")
# print(f"Output shape: {output.shape}")

# encoder = model.features          
# with torch.no_grad():
#     features = encoder(dummy_input)
# print(features.shape)

# # Hook to capture intermediate outputs
# intermediate_outputs = {}

# def get_activation(name):
#     def hook(model, input, output):
#         intermediate_outputs[name] = output.detach()
#     return hook

# # Register hooks on key layers
# # Inspect the model first to identify layer names
# print(model)

# # Example: Register hooks (adjust layer names based on actual architecture)
# model.features[3].register_forward_hook(get_activation('layer1'))
# model.features[6].register_forward_hook(get_activation('layer2'))
# model.features[13].register_forward_hook(get_activation('layer3'))
# model.features[17].register_forward_hook(get_activation('layer4'))

# # Forward pass
# with torch.no_grad():
#     output = model(dummy_input)

# # Print intermediate layer dimensions
# for name, feature_map in intermediate_outputs.items():
#     print(f"{name}: {feature_map.shape}")


# class MobileNetV2Encoder(torch.nn.Module):
#     def __init__(self, pretrained_path=None):
#         super().__init__()
#         # Load pretrained MobileNetV2
#         base_model = mobilenetv2()
#         if pretrained_path:
#             base_model.load_state_dict(torch.load(pretrained_path))
        
#         # Extract feature extraction layers
#         self.features = base_model.features
        
#         # Define skip connection points (adjust indices based on architecture)
#         self.skip_indices = [3, 6, 13, 17]
    
#     def forward(self, x):
#         skip_connections = []
        
#         for idx, layer in enumerate(self.features):
#             x = layer(x)
#             # Save features at skip connection points
#             if idx in self.skip_indices:
#                 skip_connections.append(x)
        
#         return x, skip_connections

# # Test the encoder
# encoder = MobileNetV2Encoder('pretrained/mobilenetv2-c5e733a8.pth')
# encoder.eval()

# with torch.no_grad():
#     final_output, skips = encoder(dummy_input)

# print(f"Final encoder output: {final_output.shape}")
# c=[]
# for i, skip in enumerate(skips):
#     print(f"Skip connection {i+1}: {skip.shape}")
#     c.append(skip.shape[1])


# class MyBottleNeck(nn.Module):
#     def __init__(self, input, output, exp_factor=6):
#         super(MyBottleNeck, self).__init__()
#         hidden_dim=input*exp_factor
#         self.res_available = (input == output)
#         self.conv = nn.Sequential(
#             #pointvise conv
#             nn.Conv2d(input, hidden_dim, 1, bias=False),
#             nn.BatchNorm2d(hidden_dim),
#             nn.ReLU6(inplace=True),

#             #DSC
#             nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim, bias=False),
#             nn.BatchNorm2d(hidden_dim),
#             nn.ReLU6(inplace=True),

#             #linear
#             nn.Conv2d(hidden_dim, output, 1, bias=False),
#             nn.BatchNorm2d(output),
#         )

#     def forward(self, x):
#         if self.res_available :
#             return x + self.conv(x)
#         else:
#             return self.conv(x)
        


# class MyDecoderBlock_D1(nn.Module):
#     def __init__(self, input_chan, output_chan, skip_chan):
#         super(MyDecoderBlock_D1, self).__init__()
#         self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
#         self.bottle_neck1 = MyBottleNeck(input=(input_chan//4 + skip_chan), output=output_chan)
#         self.bottle_neck2 = MyBottleNeck(input=output_chan, output=output_chan)
#         self.bottle_neck3 = MyBottleNeck(input=output_chan, output=output_chan)
#         self.bottle_neck4 = MyBottleNeck(input=output_chan, output=output_chan)

#     def forward(self, x, skip_tensor):
#         x = self.pixel_shuffle(x)
#         x = torch.cat([x, skip_tensor], dim=1)
#         x = self.bottle_neck1(x)
#         x = self.bottle_neck2(x)
#         x = self.bottle_neck3(x)
#         x = self.bottle_neck4(x)
#         return x         
    

# class MyDecoderBlock_D2(nn.Module):
#     def __init__(self, input_chan, output_chan, skip_chan):
#         super(MyDecoderBlock_D2, self).__init__()
#         self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
#         self.bottle_neck1 = MyBottleNeck(input=(input_chan//4 + skip_chan), output=output_chan)
#         self.bottle_neck2 = MyBottleNeck(input=output_chan, output=output_chan)
#         self.bottle_neck3 = MyBottleNeck(input=output_chan, output=output_chan)

#     def forward(self, x, skip_tensor):
#         x = self.pixel_shuffle(x)
#         x = torch.cat([x, skip_tensor], dim=1)
#         x = self.bottle_neck1(x)
#         x = self.bottle_neck2(x)
#         x = self.bottle_neck3(x)
#         return x         
    

# class MyDecoderBlock_D3(nn.Module):
#     def __init__(self, input_chan, output_chan, skip_chan):
#         super(MyDecoderBlock_D3, self).__init__()
#         self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
#         self.bottle_neck1 = MyBottleNeck(input=(input_chan//4 + skip_chan), output=output_chan)
#         self.bottle_neck2 = MyBottleNeck(input=output_chan, output=output_chan)
#         self.bottle_neck3 = MyBottleNeck(input=output_chan, output=output_chan)

#     def forward(self, x, skip_tensor):
#         x = self.pixel_shuffle(x)
#         x = torch.cat([x, skip_tensor], dim=1)
#         x = self.bottle_neck1(x)
#         x = self.bottle_neck2(x)
#         x = self.bottle_neck3(x)
#         return x         
    

# class MyDecoderBlock_D4(nn.Module):
#     def __init__(self, input_chan, output_chan, skip_chan):
#         super(MyDecoderBlock_D4, self).__init__()
#         self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
#         self.bottle_neck1 = MyBottleNeck(input=(input_chan//4 + skip_chan), output=output_chan)
#         self.bottle_neck2 = MyBottleNeck(input=output_chan, output=output_chan)

#     def forward(self, x, skip_tensor):
#         x = self.pixel_shuffle(x)
#         x = torch.cat([x, skip_tensor], dim=1)
#         x = self.bottle_neck1(x)
#         x = self.bottle_neck2(x)
#         return x         
    

# import torch.nn.functional as F

# class MyDecoder(nn.Module):
#     def __init__(self, MyDecoderBlock_D1, MyDecoderBlock_D2, MyDecoderBlock_D3, MyDecoderBlock_D4):
#         super(MyDecoder, self).__init__()
#         self.block1 = MyDecoderBlock_D1(input_chan=320, output_chan=96, skip_chan=c[3])
#         self.block2 = MyDecoderBlock_D2(input_chan=96, output_chan=32, skip_chan=c[2])
#         self.block3 = MyDecoderBlock_D3(input_chan=32, output_chan=24, skip_chan=c[1])
#         self.block4 = MyDecoderBlock_D4(input_chan=24, output_chan=8, skip_chan=c[0])

#     def forward(self, x, skip1, skip2, skip3, skip4):
#         if x.shape[2:] != skip1.shape[2:]:
#             skip1 = F.interpolate(skip1, size=x.shape[2:], mode='bilinear', align_corners=False)
#         x = self.block1(x, skip1)
        
#         if x.shape[2:] != skip2.shape[2:]:
#             skip2 = F.interpolate(skip2, size=x.shape[2:], mode='bilinear', align_corners=False)
#         x = self.block2(x, skip2)
        
#         if x.shape[2:] != skip3.shape[2:]:
#             skip3 = F.interpolate(skip3, size=x.shape[2:], mode='bilinear', align_corners=False)
#         x = self.block3(x, skip3)
        
#         if x.shape[2:] != skip4.shape[2:]:
#             skip4 = F.interpolate(skip4, size=x.shape[2:], mode='bilinear', align_corners=False)
#         x = self.block4(x, skip4)
        
#         return x



# class MyModel(nn.Module):
#     def __init__(self, Encoder, Decoder):
#         super(MyModel, self).__init__()
#         self.encoder = Encoder
#         self.decoder = Decoder

#     def forward(self, x):
#         x, skips = self.encoder(x)
#         output = self.decoder(x, *skips)
#         return output
    

# encoder = MobileNetV2Encoder(pretrained_path='pretrained/mobilenetv2-c5e733a8.pth')
# decoder = MyDecoder(MyDecoderBlock_D1, MyDecoderBlock_D2, MyDecoderBlock_D3, MyDecoderBlock_D4)

# model = MyModel(encoder, decoder)


# from PIL import Image
# import torch
# from torchvision import transforms

# transform = transforms.Compose([
#     transforms.Resize((224,224)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], 
#                          [0.229, 0.224, 0.225]),
# ])
# img = Image.open('ph.jpg')
# img_tensor = transform(img).unsqueeze(0)  # [1, 3, 224, 224]

# output = model(img_tensor)  # shape depends on model

# print(type(output), output.shape)
# print(output[0,0,:5,:5])  # show slice for inspection










# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision import transforms
# from PIL import Image
# import matplotlib.pyplot as plt

# from models.imagenet import mobilenetv2
# model = mobilenetv2()
# dummy_input = torch.randn(1, 3, 224, 224)

# class MobileNetV2Encoder(torch.nn.Module):
#     def __init__(self, pretrained_path=None):
#         super().__init__()
#         base_model = mobilenetv2()
#         if pretrained_path:
#             base_model.load_state_dict(torch.load(pretrained_path))
#         self.features = base_model.features
#         self.skip_indices = [3, 6, 13, 17]

#     def forward(self, x):
#         skip_connections = []
#         for idx, layer in enumerate(self.features):
#             x = layer(x)
#             if idx in self.skip_indices:
#                 skip_connections.append(x)
#         return x, skip_connections



# encoder = MobileNetV2Encoder(pretrained_path='pretrained/mobilenetv2-c5e733a8.pth')
# encoder.eval()

# with torch.no_grad():
#     final_output, skips = encoder(dummy_input)

# c = [skip.shape[1] for skip in skips]
# print("Skip channels:", c)

# class MyBottleNeck(nn.Module):
#     def __init__(self, input, output, exp_factor=6):
#         super(MyBottleNeck, self).__init__()
#         hidden_dim=input*exp_factor
#         self.res_available = (input == output)
#         self.conv = nn.Sequential(
#             #pointvise conv
#             nn.Conv2d(input, hidden_dim, 1, bias=False),
#             nn.BatchNorm2d(hidden_dim),
#             nn.ReLU6(inplace=True),

#             #DSC
#             nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim, bias=False),
#             nn.BatchNorm2d(hidden_dim),
#             nn.ReLU6(inplace=True),

#             #linear
#             nn.Conv2d(hidden_dim, output, 1, bias=False),
#             nn.BatchNorm2d(output),
#         )

#     def forward(self, x):
#         if self.res_available :
#             return x + self.conv(x)
#         else:
#             return self.conv(x)
        
# class MyDecoderBlock_D1(nn.Module):
#     def __init__(self, input_chan, output_chan, skip_chan):
#         super(MyDecoderBlock_D1, self).__init__()
#         self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
#         self.bottle_neck1 = MyBottleNeck(input=(input_chan//4 + skip_chan), output=output_chan)
#         self.bottle_neck2 = MyBottleNeck(input=output_chan, output=output_chan)
#         self.bottle_neck3 = MyBottleNeck(input=output_chan, output=output_chan)
#         self.bottle_neck4 = MyBottleNeck(input=output_chan, output=output_chan)

#     def forward(self, x, skip_tensor):
#         x = self.pixel_shuffle(x)

#         print(f'[D1] x shape: {x.shape}, skip shape: {skip_tensor.shape}')

#         if x.shape[2:] != skip_tensor.shape[2:]:
#             if x.shape[2] < skip_tensor.shape[2]:
#                 x = F.interpolate(x, size=skip_tensor.shape[2:], mode='bilinear', align_corners=False)
#             else:
#                 skip_tensor = F.interpolate(skip_tensor, size=x.shape[2:], mode='bilinear', align_corners=False)

#         x = torch.cat([x, skip_tensor], dim=1)
#         x = self.bottle_neck1(x)
#         x = self.bottle_neck2(x)
#         x = self.bottle_neck3(x)
#         x = self.bottle_neck4(x)
#         return x         


# class MyDecoderBlock_D2(nn.Module):
#     def __init__(self, input_chan, output_chan, skip_chan):
#         super(MyDecoderBlock_D2, self).__init__()
#         self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
#         self.bottle_neck1 = MyBottleNeck(input=(input_chan//4 + skip_chan), output=output_chan)
#         self.bottle_neck2 = MyBottleNeck(input=output_chan, output=output_chan)
#         self.bottle_neck3 = MyBottleNeck(input=output_chan, output=output_chan)

#     def forward(self, x, skip_tensor):
#         x = self.pixel_shuffle(x)

#         print(f'[D2] x shape: {x.shape}, skip shape: {skip_tensor.shape}')

#         if x.shape[2:] != skip_tensor.shape[2:]:
#             if x.shape[2] < skip_tensor.shape[2]:
#                 x = F.interpolate(x, size=skip_tensor.shape[2:], mode='bilinear', align_corners=False)
#             else:
#                 skip_tensor = F.interpolate(skip_tensor, size=x.shape[2:], mode='bilinear', align_corners=False)

#         x = torch.cat([x, skip_tensor], dim=1)
#         x = self.bottle_neck1(x)
#         x = self.bottle_neck2(x)
#         x = self.bottle_neck3(x)
#         return x         


# class MyDecoderBlock_D3(nn.Module):
#     def __init__(self, input_chan, output_chan, skip_chan):
#         super(MyDecoderBlock_D3, self).__init__()
#         self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
#         self.bottle_neck1 = MyBottleNeck(input=(input_chan//4 + skip_chan), output=output_chan)
#         self.bottle_neck2 = MyBottleNeck(input=output_chan, output=output_chan)
#         self.bottle_neck3 = MyBottleNeck(input=output_chan, output=output_chan)

#     def forward(self, x, skip_tensor):
#         x = self.pixel_shuffle(x)

#         print(f'[D3] x shape: {x.shape}, skip shape: {skip_tensor.shape}')

#         if x.shape[2:] != skip_tensor.shape[2:]:
#             if x.shape[2] < skip_tensor.shape[2]:
#                 x = F.interpolate(x, size=skip_tensor.shape[2:], mode='bilinear', align_corners=False)
#             else:
#                 skip_tensor = F.interpolate(skip_tensor, size=x.shape[2:], mode='bilinear', align_corners=False)

#         x = torch.cat([x, skip_tensor], dim=1)
#         x = self.bottle_neck1(x)
#         x = self.bottle_neck2(x)
#         x = self.bottle_neck3(x)
#         return x         


# class MyDecoderBlock_D4(nn.Module):
#     def __init__(self, input_chan, output_chan, skip_chan):
#         super(MyDecoderBlock_D4, self).__init__()
#         self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
#         self.bottle_neck1 = MyBottleNeck(input=(input_chan//4 + skip_chan), output=output_chan)
#         self.bottle_neck2 = MyBottleNeck(input=output_chan, output=output_chan)

#     def forward(self, x, skip_tensor):
#         x = self.pixel_shuffle(x)

#         print(f'[D4] x shape: {x.shape}, skip shape: {skip_tensor.shape}')

#         if x.shape[2:] != skip_tensor.shape[2:]:
#             if x.shape[2] < skip_tensor.shape[2]:
#                 x = F.interpolate(x, size=skip_tensor.shape[2:], mode='bilinear', align_corners=False)
#             else:
#                 skip_tensor = F.interpolate(skip_tensor, size=x.shape[2:], mode='bilinear', align_corners=False)

#         x = torch.cat([x, skip_tensor], dim=1)
#         x = self.bottle_neck1(x)
#         x = self.bottle_neck2(x)
#         return x         



# class MyDecoder(nn.Module):
#     def __init__(self, MyDecoderBlock_D1, MyDecoderBlock_D2, MyDecoderBlock_D3, MyDecoderBlock_D4):
#         super(MyDecoder, self).__init__()
#         self.block1 = MyDecoderBlock_D1(input_chan=320, output_chan=96, skip_chan=c[3])
#         self.block2 = MyDecoderBlock_D2(input_chan=96, output_chan=32, skip_chan=c[2])
#         self.block3 = MyDecoderBlock_D3(input_chan=32, output_chan=24, skip_chan=c[1])
#         self.block4 = MyDecoderBlock_D4(input_chan=24, output_chan=8, skip_chan=c[0])

#     def forward(self, x, skip1, skip2, skip3, skip4):  
#         x = self.block1(x, skip1)
#         x = self.block2(x, skip2)
#         x = self.block3(x, skip3)
#         x = self.block4(x, skip4)
#         return x
    

# decoder = MyDecoder(MyDecoderBlock_D1, MyDecoderBlock_D2, MyDecoderBlock_D3, MyDecoderBlock_D4)
# class MyModel(nn.Module):
#     def __init__(self, Encoder, Decoder):
#         super(MyModel, self).__init__()
#         self.encoder = Encoder
#         self.decoder = Decoder

#     def forward(self, x):
#         x, skips = self.encoder(x)
#         output = self.decoder(x, *skips[::-1])
#         return output
    
# model = MyModel(encoder, decoder)



import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import os
import numpy as np


#loading the mobilenetv2 model
from models.imagenet import mobilenetv2
model = mobilenetv2()
dummy_input = torch.randn(1, 3, 1056, 320)

#wrapping mobilenetv2 as the encoder and loading its pretrained weights
class MobileNetV2Encoder(torch.nn.Module):
    def __init__(self, pretrained_path=None):
        super().__init__()
        base_model = mobilenetv2()
        if pretrained_path:
            base_model.load_state_dict(torch.load(pretrained_path))
        self.features = base_model.features
        self.skip_indices = [3, 6, 13, 17]

    def forward(self, x):
        skip_connections = []
        for idx, layer in enumerate(self.features):
            x = layer(x)
            if idx in self.skip_indices:
                skip_connections.append(x)
        return x, skip_connections



encoder = MobileNetV2Encoder(pretrained_path='pretrained/mobilenetv2-c5e733a8.pth')
encoder.eval()

#runs the encoder on dummy_input to get skip channel sizes
with torch.no_grad():
    final_output, skips = encoder(dummy_input)

c = [skip.shape[1] for skip in skips]
print("Skip channels:", c)
print(f"Final encoder output: {final_output.shape}")
c=[]
for i, skip in enumerate(skips):
    print(f"Skip connection {i+1}: {skip.shape}")
    c.append(skip.shape[1])

intermediate_outputs = {}

def get_activation(name):
    def hook(model, input, output):
        intermediate_outputs[name] = output.detach()
    return hook

# Register hooks on key layers
# Inspect the model first to identify layer names
print(model)

# Example: Register hooks (adjust layer names based on actual architecture)
model.features[3].register_forward_hook(get_activation('layer1'))
model.features[6].register_forward_hook(get_activation('layer2'))
model.features[13].register_forward_hook(get_activation('layer3'))
model.features[17].register_forward_hook(get_activation('layer4'))

# Forward pass
with torch.no_grad():
    output = model(dummy_input)

# Print intermediate layer dimensions
for name, feature_map in intermediate_outputs.items():
    print(f"{name}: {feature_map.shape}")

class MyBottleNeck(nn.Module):
    def __init__(self, input, output, exp_factor=6):
        super(MyBottleNeck, self).__init__()
        hidden_dim=input*exp_factor
        self.res_available = (input == output)
        self.conv = nn.Sequential(
            #pointvise conv
            nn.Conv2d(input, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),

            #DSC
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),

            #linear
            nn.Conv2d(hidden_dim, output, 1, bias=False),
            nn.BatchNorm2d(output),
        )

    def forward(self, x):
        if self.res_available :
            return x + self.conv(x)
        else:
            return self.conv(x)
        

class MyDecoderBlock_D1(nn.Module):
    def __init__(self, input_chan, output_chan, skip_chan):
        super(MyDecoderBlock_D1, self).__init__()
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
        self.bottle_neck1 = MyBottleNeck(input=(input_chan+skip_chan)//4, output=output_chan)
        self.bottle_neck2 = MyBottleNeck(input=output_chan, output=output_chan)
        self.bottle_neck3 = MyBottleNeck(input=output_chan, output=output_chan)
        self.bottle_neck4 = MyBottleNeck(input=output_chan, output=output_chan)

    def forward(self, x, skip_tensor):
        x = torch.cat([x, skip_tensor], dim=1)
        x = self.pixel_shuffle(x)
        x = self.bottle_neck1(x)
        x = self.bottle_neck2(x)
        x = self.bottle_neck3(x)
        x = self.bottle_neck4(x)
        return x         


class MyDecoderBlock_D2(nn.Module):
    def __init__(self, input_chan, output_chan, skip_chan):
        super(MyDecoderBlock_D2, self).__init__()
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
        self.bottle_neck1 = MyBottleNeck(input=(input_chan + skip_chan)//4, output=output_chan)
        self.bottle_neck2 = MyBottleNeck(input=output_chan, output=output_chan)
        self.bottle_neck3 = MyBottleNeck(input=output_chan, output=output_chan)

    def forward(self, x, skip_tensor):
        x = torch.cat([x, skip_tensor], dim=1)
        x = self.pixel_shuffle(x)
        x = self.bottle_neck1(x)
        x = self.bottle_neck2(x)
        x = self.bottle_neck3(x)
        return x         

class MyDecoderBlock_D3(nn.Module):
    def __init__(self, input_chan, output_chan, skip_chan):
        super(MyDecoderBlock_D3, self).__init__()
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
        self.bottle_neck1 = MyBottleNeck(input=(input_chan + skip_chan)//4, output=output_chan)
        self.bottle_neck2 = MyBottleNeck(input=output_chan, output=output_chan)
        self.bottle_neck3 = MyBottleNeck(input=output_chan, output=output_chan)

    def forward(self, x, skip_tensor):
        x = torch.cat([x, skip_tensor], dim=1)
        x = self.pixel_shuffle(x)
        x = self.bottle_neck1(x)
        x = self.bottle_neck2(x)
        x = self.bottle_neck3(x)
        return x         

class MyDecoderBlock_D4(nn.Module):
    def __init__(self, input_chan, output_chan, skip_chan):
        super(MyDecoderBlock_D4, self).__init__()
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=4)
        self.bottle_neck1 = MyBottleNeck(input=(input_chan + skip_chan)//4, output=output_chan)
        self.bottle_neck2 = MyBottleNeck(input=output_chan, output=output_chan)

    def forward(self, x, skip_tensor):
        x = torch.cat([x, skip_tensor], dim=1)        
        x = self.pixel_shuffle(x)
        x = self.bottle_neck1(x)
        x = self.bottle_neck2(x)
        return x         



class MyDecoder(nn.Module):
    def __init__(self, MyDecoderBlock_D1, MyDecoderBlock_D2, MyDecoderBlock_D3, MyDecoderBlock_D4):
        super(MyDecoder, self).__init__()
        self.block1 = MyDecoderBlock_D1(input_chan=320, output_chan=96, skip_chan=c[3])
        self.block2 = MyDecoderBlock_D2(input_chan=96, output_chan=32, skip_chan=c[2])
        self.block3 = MyDecoderBlock_D3(input_chan=32, output_chan=24, skip_chan=c[1])
        self.block4 = MyDecoderBlock_D4(input_chan=24, output_chan=8, skip_chan=c[0])

    def forward(self, x, skip1, skip2, skip3, skip4):  
        x = self.block1(x, skip1)
        x = self.block2(x, skip2)
        x = self.block3(x, skip3)
        x = self.block4(x, skip4)
        return x
    
decoder = MyDecoder(MyDecoderBlock_D1, MyDecoderBlock_D2, MyDecoderBlock_D3, MyDecoderBlock_D4)

class MyModel(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(MyModel, self).__init__()
        self.encoder = Encoder
        self.decoder = Decoder

    def forward(self, x):
        x, skips = self.encoder(x)
        output = self.decoder(x, *skips[::-1])
        return output
    
model = MyModel(encoder, decoder)

transform = transforms.Compose([
    transforms.Resize((1056,320)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225]),
])

img = Image.open('ph.jpg')
img_tensor = transform(img).unsqueeze(0)

# Forward pass
print(img_tensor.shape)
output = model(img_tensor)
print(type(output), output.shape)

# LOSS

alpha=.25
beta=.75
m=4
M=80

def norm_log_transform(depthmap):
    depthmap_clamp = torch.clamp(depthmap, min=m, max=M)
    log_trans = torch.log(depthmap_clamp)
    log_m = torch.log(torch.tensor(m, dtype=depthmap.dtype, device=depthmap.device))
    log_M = torch.log(torch.tensor(M, dtype=depthmap.dtype, device=depthmap.device))

    g_d = ((log_trans - log_m) * M) / (log_M - log_m)
    return g_d


def berhu_loss(depthmap_grnd, depthmap_pred):
    diff = depthmap_grnd - depthmap_pred 
    abs_diff = torch.abs(diff)
    c = torch.max(abs_diff).item() / 5 
    
    loss = torch.where(abs_diff <= c, abs_diff, (diff ** 2 + c ** 2) / (2 * c))
    return loss.mean()

def depth_loss(depthmap_grnd, depthmap_pred):
    norm_depthmap_grnd=norm_log_transform(depthmap=depthmap_grnd)
    norm_depthmap_pred=norm_log_transform(depthmap=depthmap_pred)
    loss = berhu_loss(norm_depthmap_grnd, norm_depthmap_pred)
    return loss


def segmentation_loss(logits, labels):
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, labels)
    return loss

def total_loss(depthmap_grnds, depthmap_preds, segmentmap_grnd, segmentmap_preds):
    loss = 0
    for depthmap_grnd, depthmap_pred, segmentmap_grnd, segmentmap_pred in zip(depthmap_grnds, depthmap_preds, segmentmap_grnd, segmentmap_preds):
        loss += (alpha * (depth_loss(depthmap_grnd=depthmap_grnd, depthmap_pred=depthmap_pred)) + beta * (segmentation_loss(logits=segmentmap_pred, label=segmentmap_grnd)))
    return loss


class CustomDataset(Dataset):
    def __init__(self, img_dir, depthmap_dir, segmentmap_dir,
                 transform_img_dir=None,
                 transform_depthmap_dir=None,
                 transform_segmentmap_dir=None):
        self.img_dir = img_dir
        self.depthmap_dir = depthmap_dir
        self.segmentmap_dir = segmentmap_dir
        self.transform_img_dir = transform_img_dir
        self.transform_depthmap_dir = transform_depthmap_dir
        self.transform_segmentmap_dir = transform_segmentmap_dir
        self.img_dir_filenames = sorted([f for f in os.listdir(img_dir) if f.endswith('.npy')])

    def __len__(self):
        return len(self.img_dir_filenames)
    
    def __getitem__(self, idx):
        img_name = self.img_dir_filenames[idx]
        img_path = os.path.join(self.img_dir, img_name)
        depth_path = os.path.join(self.depthmap_dir, img_name)  # same naming
        seg_path = os.path.join(self.segmentmap_dir, img_name)

        img = np.load(img_path) 
        depthmap = np.load(depth_path)
        segmentmap = np.load(seg_path)

        # Convert to torch tensors
        if img.ndim == 3:
            img = torch.from_numpy(img).permute(2, 0, 1).float()  # CHW (channels first)
        else:
            img = torch.from_numpy(img).float()
        depthmap = torch.from_numpy(depthmap).float()  # usually single channel
        segmentmap = torch.from_numpy(segmentmap).long()  # for class indices

        # You can add normalization or resizing for tensors as needed here,
        # or preprocess before saving your .npy files.
        
        return img, depthmap, segmentmap

train_img_dir = './data/train/image'
train_depthmap_dir = './data/train/depth'
train_segmentmap_dir = './data/train/label'

val_img_dir = './data/val/image'
val_depthmap_dir = './data/val/depth'
val_segmentmap_dir = './data/val/label'

train_Dataset = CustomDataset(train_img_dir, train_depthmap_dir, train_segmentmap_dir)
train_Loader = DataLoader(train_Dataset, batch_size=8, shuffle=True, num_workers=4)


#training loop:
epochs=20
itr=1000
batch_size=4
learning_rate = .01
model.train()

device = 'cpu'
model = model.to('cpu')
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  
scheduler = StepLR(optimizer, step_size=1, gamma=0.90)  

for epoch in range(epochs):
    running_loss = 0.0
    for i, (imgs, depths, segs) in enumerate(train_Loader):
        if i >= itr:
            break
        imgs = imgs.to(device)
        depths = depths.to(device)
        segs = segs.to(device)
        depths = depths.to(device).squeeze(-1)

        optimizer.zero_grad(set_to_none=True)
        pred_depths, pred_segs = model(imgs)
        pred_depths = pred_depths.squeeze(1)

        print('pred_depths:', pred_depths.shape)
        print('depths:', depths.shape)
        print(torch.min(segs), torch.max(segs))
        print(pred_segs.shape)
        print(segs.shape)

        loss_total = total_loss(depths, pred_depths, segs, pred_segs)
        loss_total.backward()

        # Check gradients for NaN or inf
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    print(f"NaN gradient detected in {name}")
                    raise ValueError("NaN in gradients, stopping")
                if torch.isinf(param.grad).any():
                    print(f"Inf gradient detected in {name}")
                    raise ValueError("Inf in gradients, stopping")

        optimizer.step()

        loss_total_value = loss_total.item()
        assert not torch.isnan(torch.tensor(loss_total_value)), "NaN loss"
    avg_loss = running_loss / itr
    print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.4f}")
    scheduler.step()
    print(f"Learning rate after epoch {epoch+1}: {optimizer.param_groups[0]['lr']}")









import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import os
import numpy as np



#loading the mobilenetv2 model
from models.imagenet import mobilenetv2
model = mobilenetv2()
dummy_input = torch.randn(1, 3, 1056, 320)

#wrapping mobilenetv2 as the encoder and loading its pretrained weights
class MobileNetV2Encoder(torch.nn.Module):
    def __init__(self, pretrained_path=None):
        super().__init__()
        base_model = mobilenetv2()
        if pretrained_path:
            base_model.load_state_dict(torch.load(pretrained_path))
        self.features = base_model.features
        self.skip_indices = [3, 6, 13, 17]

    def forward(self, x):
        skip_connections = []
        for idx, layer in enumerate(self.features):
            x = layer(x)
            if idx in self.skip_indices:
                skip_connections.append(x)
        return x, skip_connections



encoder = MobileNetV2Encoder(pretrained_path='pretrained/mobilenetv2-c5e733a8.pth')
encoder.eval()

#runs the encoder on dummy_input to get skip channel sizes
with torch.no_grad():
    final_output, skips = encoder(dummy_input)

c = [skip.shape[1] for skip in skips]
print("Skip channels:", c)
print(f"Final encoder output: {final_output.shape}")
c=[]
for i, skip in enumerate(skips):
    print(f"Skip connection {i+1}: {skip.shape}")
    c.append(skip.shape[1])

intermediate_outputs = {}

def get_activation(name):
    def hook(model, input, output):
        intermediate_outputs[name] = output.detach()
    return hook

# Register hooks on key layers
# Inspect the model first to identify layer names
print(model)

# Example: Register hooks (adjust layer names based on actual architecture)
model.features[3].register_forward_hook(get_activation('layer1'))
model.features[6].register_forward_hook(get_activation('layer2'))
model.features[13].register_forward_hook(get_activation('layer3'))
model.features[17].register_forward_hook(get_activation('layer4'))

# Forward pass
with torch.no_grad():
    output = model(dummy_input)

# Print intermediate layer dimensions
for name, feature_map in intermediate_outputs.items():
    print(f"{name}: {feature_map.shape}")

class MyBottleNeck(nn.Module):
    def __init__(self, input, output, exp_factor=6):
        super(MyBottleNeck, self).__init__()
        hidden_dim=input*exp_factor
        self.res_available = (input == output)
        self.conv = nn.Sequential(
            #pointvise conv
            nn.Conv2d(input, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),

            #DSC
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),

            #linear
            nn.Conv2d(hidden_dim, output, 1, bias=False),
            nn.BatchNorm2d(output),
        )

    def forward(self, x):
        if self.res_available :
            return x + self.conv(x)
        else:
            return self.conv(x)

class MyDecoderBlock_D1(nn.Module):
    def __init__(self, input_chan, output_chan, skip_chan):
        super(MyDecoderBlock_D1, self).__init__()
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
        self.bottle_neck1 = MyBottleNeck(input=(input_chan+skip_chan)//4, output=output_chan)
        self.bottle_neck2 = MyBottleNeck(input=output_chan, output=output_chan)
        self.bottle_neck3 = MyBottleNeck(input=output_chan, output=output_chan)
        self.bottle_neck4 = MyBottleNeck(input=output_chan, output=output_chan)

    def forward(self, x, skip_tensor):
        x = torch.cat([x, skip_tensor], dim=1)
        print(f"Shape after concatenation: {x.shape}")
        x = self.pixel_shuffle(x)
        print(f"Shape after pixel shuffle: {x.shape}")
        x = self.bottle_neck1(x)
        x = self.bottle_neck2(x)
        x = self.bottle_neck3(x)
        x = self.bottle_neck4(x)
        return x
    

with torch.no_grad():
    encoder_output, skip_connections = encoder(dummy_input)
skip_for_D1 = skip_connections[-1]
decoder_block_D1 = MyDecoderBlock_D1(input_chan=320, output_chan=96, skip_chan=skip_for_D1.shape[1])

print("Before forward:")
print(f"Encoder output shape: {encoder_output.shape}")
print(f"Skip connection shape: {skip_for_D1.shape}")
output_D1 = decoder_block_D1(encoder_output, skip_for_D1)
print("After forward:")
print(f"Output shape: {output_D1.shape}")


class MyDecoderBlock_D2(nn.Module):
    def __init__(self, input_chan, output_chan, skip_chan):
        super(MyDecoderBlock_D2, self).__init__()
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
        self.bottle_neck1 = MyBottleNeck(input=(input_chan + skip_chan)//4, output=output_chan)
        self.bottle_neck2 = MyBottleNeck(input=output_chan, output=output_chan)
        self.bottle_neck3 = MyBottleNeck(input=output_chan, output=output_chan)

    def forward(self, x, skip_tensor):
        print(f'[D2] x shape: {x.shape}, skip shape: {skip_tensor.shape}')

        # if x.shape[2:] != skip_tensor.shape[2:]:
        #     if x.shape[2] < skip_tensor.shape[2]:
        #         x = F.interpolate(x, size=skip_tensor.shape[2:], mode='bilinear', align_corners=False)
        #     else:
        #         skip_tensor = F.interpolate(skip_tensor, size=x.shape[2:], mode='bilinear', align_corners=False)

        x = torch.cat([x, skip_tensor], dim=1)
        print(f"Shape after concatenation: {x.shape}")
        x = self.pixel_shuffle(x)
        print(f"Shape after pixel shuffle: {x.shape}")
        x = self.bottle_neck1(x)
        x = self.bottle_neck2(x)
        x = self.bottle_neck3(x)
        return x         

with torch.no_grad():
    encoder_output, skip_connections = encoder(dummy_input)

skip_for_D2 = skip_connections[-2]
decoder_block_D2 = MyDecoderBlock_D2(input_chan=96, output_chan=32, skip_chan=skip_for_D2.shape[1])  

print("Before forward D2:")
print(f"Input tensor shape: {output_D1.shape}")  # output of D1 is input to D2
print(f"Skip connection shape: {skip_for_D2.shape}")

output_D2 = decoder_block_D2(output_D1, skip_for_D2)

print("After forward D2:")
print(f"Output shape: {output_D2.shape}")

class MyDecoderBlock_D3(nn.Module):
    def __init__(self, input_chan, output_chan, skip_chan):
        super(MyDecoderBlock_D3, self).__init__()
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
        self.bottle_neck1 = MyBottleNeck(input=(input_chan + skip_chan)//4, output=output_chan)
        self.bottle_neck2 = MyBottleNeck(input=output_chan, output=output_chan)
        self.bottle_neck3 = MyBottleNeck(input=output_chan, output=output_chan)

    def forward(self, x, skip_tensor):
        x = torch.cat([x, skip_tensor], dim=1)
        print(f"Shape after concatenation: {x.shape}")
        x = self.pixel_shuffle(x)
        print(f"Shape after pixel shuffle: {x.shape}")
        x = self.bottle_neck1(x)
        x = self.bottle_neck2(x)
        x = self.bottle_neck3(x)
        return x   
          
with torch.no_grad():
    encoder_output, skip_connections = encoder(dummy_input)

skip_for_D3 = skip_connections[-3]
decoder_block_D3 = MyDecoderBlock_D3(input_chan=32, output_chan=24, skip_chan=skip_for_D3.shape[1])  

print("Before forward D3:")
print(f"Input tensor shape: {output_D2.shape}")  # output of D1 is input to D2
print(f"Skip connection shape: {skip_for_D3.shape}")

output_D3 = decoder_block_D3(output_D2, skip_for_D3)

print("After forward D3:")
print(f"Output shape: {output_D3.shape}")

class MyDecoderBlock_D4(nn.Module):
    def __init__(self, input_chan, output_chan, skip_chan):
        super(MyDecoderBlock_D4, self).__init__()
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=4)
        self.bottle_neck1 = MyBottleNeck(input=(input_chan + skip_chan)//16, output=output_chan)
        self.bottle_neck2 = MyBottleNeck(input=output_chan, output=output_chan)

    def forward(self, x, skip_tensor):
        x = torch.cat([x, skip_tensor], dim=1)
        print(f"Shape after concatenation: {x.shape}")
        x = self.pixel_shuffle(x)
        print(f"Shape after pixel shuffle: {x.shape}")
        x = self.bottle_neck1(x)
        x = self.bottle_neck2(x)
        return x         


skip_for_D4 = skip_connections[-4]

decoder_block_D4 = MyDecoderBlock_D4(input_chan=24, output_chan=8, skip_chan=skip_for_D4.shape[1])

print("Before forward D4:")
print(f"Input tensor shape: {output_D3.shape}")  # output of D3 is input to D4
print(f"Skip connection shape: {skip_for_D4.shape}")

output_D4 = decoder_block_D4(output_D3, skip_for_D4)

print("After forward D4:")
print(f"Output shape: {output_D4.shape}")



class MyDecoder(nn.Module):
    def __init__(self, MyDecoderBlock_D1, MyDecoderBlock_D2, MyDecoderBlock_D3, MyDecoderBlock_D4):
        super(MyDecoder, self).__init__()
        self.block1 = MyDecoderBlock_D1(input_chan=320, output_chan=96, skip_chan=c[3])
        self.block2 = MyDecoderBlock_D2(input_chan=96, output_chan=32, skip_chan=c[2])
        self.block3 = MyDecoderBlock_D3(input_chan=32, output_chan=24, skip_chan=c[1])
        self.block4 = MyDecoderBlock_D4(input_chan=24, output_chan=8, skip_chan=c[0])
        self.depth_head = nn.Conv2d(8, 1, 1)   
        self.segmentation_head = nn.Conv2d(8, 19, 1) 

    def forward(self, x, skip1, skip2, skip3, skip4):  
        x = self.block1(x, skip1)
        x = self.block2(x, skip2)
        x = self.block3(x, skip3)
        x = self.block4(x, skip4)
        depth = self.depth_head(x)
        seg = self.segmentation_head(x)
        return depth, seg
    
decoder = MyDecoder(MyDecoderBlock_D1, MyDecoderBlock_D2, MyDecoderBlock_D3, MyDecoderBlock_D4)

class MyModel(nn.Module):
    def __init__(self, Encoder, Decoder):
        super().__init__()
        self.encoder = Encoder
        self.decoder = Decoder

    def forward(self, x):
        x, skips = self.encoder(x)
        return self.decoder(x, *skips[::-1]) 

model = MyModel(encoder, decoder)

alpha = 0.25
beta = 0.75
m = 4
M = 80

def norm_log_transform(depthmap):
    depthmap_clamp = torch.clamp(depthmap, min=m, max=M)
    log_trans = torch.log(depthmap_clamp)
    log_m = torch.log(torch.tensor(m, dtype=depthmap.dtype, device=depthmap.device))
    log_M = torch.log(torch.tensor(M, dtype=depthmap.dtype, device=depthmap.device))
    g_d = ((log_trans - log_m) * M) / (log_M - log_m)
    return g_d

def berhu_loss(depthmap_grnd, depthmap_pred):
    diff = depthmap_grnd - depthmap_pred
    abs_diff = torch.abs(diff)
    c = torch.max(abs_diff).item() / 5
    loss = torch.where(abs_diff <= c, abs_diff, (diff ** 2 + c ** 2) / (2 * c))
    return loss.mean()

def depth_loss(depthmap_grnd, depthmap_pred):
    norm_depthmap_grnd = norm_log_transform(depthmap=depthmap_grnd)
    norm_depthmap_pred = norm_log_transform(depthmap=depthmap_pred)
    return berhu_loss(norm_depthmap_grnd, norm_depthmap_pred)

def segmentation_loss(logits, labels):
    criterion = nn.CrossEntropyLoss(ignore_index=-1)  # use correct ignore index as per your dataset
    loss = criterion(logits, labels)
    return loss

def total_loss(depthmap_grnds, depthmap_preds, segmentmap_grnd, segmentmap_preds):
    # Compute batch loss directly without looping over batch frequently
    loss_depth = depth_loss(depthmap_grnds, depthmap_preds)
    loss_seg = segmentation_loss(segmentmap_preds, segmentmap_grnd)
    return alpha * loss_depth + beta * loss_seg


class CustomDataset(Dataset):
    def __init__(self, img_dir, depthmap_dir, segmentmap_dir,
                 transform_img_dir=None,
                 transform_depthmap_dir=None,
                 transform_segmentmap_dir=None):
        self.img_dir = img_dir
        self.depthmap_dir = depthmap_dir
        self.segmentmap_dir = segmentmap_dir
        self.transform_img_dir = transform_img_dir
        self.transform_depthmap_dir = transform_depthmap_dir
        self.transform_segmentmap_dir = transform_segmentmap_dir
        self.img_dir_filenames = sorted([f for f in os.listdir(img_dir) if f.endswith('.npy')])

    def __len__(self):
        return len(self.img_dir_filenames)

    def __getitem__(self, idx):
        img_name = self.img_dir_filenames[idx]
        img_path = os.path.join(self.img_dir, img_name)
        depth_path = os.path.join(self.depthmap_dir, img_name)
        seg_path = os.path.join(self.segmentmap_dir, img_name)

        img = np.load(img_path)
        depthmap = np.load(depth_path)
        segmentmap = np.load(seg_path)

        # Convert to torch tensors
        if img.ndim == 3:
            img = torch.from_numpy(img).permute(2, 0, 1).float()
        else:
            img = torch.from_numpy(img).float()
        depthmap = torch.from_numpy(depthmap).float()
        segmentmap = torch.from_numpy(segmentmap).long()

        # Return without additional transform for now
        return img, depthmap, segmentmap


train_img_dir = './data/train/image'
train_depthmap_dir = './data/train/depth'
train_segmentmap_dir = './data/train/label'

train_Dataset = CustomDataset(train_img_dir, train_depthmap_dir, train_segmentmap_dir)
train_Loader = DataLoader(train_Dataset, batch_size=4, shuffle=True, num_workers=4)

import matplotlib.pyplot as plt

def show_sample(img, depth, seg):
    img = img.detach().cpu()
    depth = depth.detach().cpu()
    seg = seg.detach().cpu()

    plt.figure(figsize=(12,4))

    # Image
    plt.subplot(1,3,1)
    plt.title('Image')
    plt.imshow(img.permute(1,2,0).numpy())

    # Depth
    plt.subplot(1,3,2)
    plt.title('Depth')
    plt.imshow(depth.squeeze().numpy(), cmap='plasma')

    # Segmentation
    plt.subplot(1,3,3)
    plt.title('Segmentation')
    plt.imshow(seg.squeeze().numpy(), cmap='tab20')

    plt.show()

# Show one sample
for imgs, depths, segs in train_Loader:
    show_sample(imgs[0], depths[0], segs[0])

    break




import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

device = torch.device('cpu')
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = StepLR(optimizer, step_size=1, gamma=0.90)

epochs = 20
itr = 1000

model.train()

for epoch in range(epochs):
    running_loss = 0.0
    for i, (imgs, depths, segs) in enumerate(train_Loader):
        if i >= itr:
            break
        imgs = imgs.to(device)
        depths = depths.to(device).squeeze(-1)
        segs = segs.to(device)

        optimizer.zero_grad(set_to_none=True)
        pred_depths, pred_segs = model(imgs)
        pred_depths = pred_depths.squeeze(1)

        loss_total = total_loss(depths, pred_depths, segs, pred_segs)
        loss_total.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Check gradients for NaNs/Infs
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    raise ValueError(f"NaN gradient detected in {name}")
                if torch.isinf(param.grad).any():
                    raise ValueError(f"Inf gradient detected in {name}")

        optimizer.step()

        running_loss += loss_total.item()

    avg_loss = running_loss / itr
    print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.4f}")
    scheduler.step()
    print(f"Learning rate after epoch {epoch+1}: {optimizer.param_groups[0]['lr']}")