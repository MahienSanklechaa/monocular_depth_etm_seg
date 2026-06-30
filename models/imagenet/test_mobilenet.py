import torch
from models.imagenet import mobilenetv2

def main():
    # Initialize model
    net = mobilenetv2()
    
    # Load pretrained weights
    net.load_state_dict(torch.load('pretrained/mobilenetv2_1.0-0c6065bc.pth', map_location='cpu'))
    net.eval()
    
    print("Model loaded successfully")

    # Dummy input tensor (batch_size=1, RGB image of 224x224)
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Forward pass
    with torch.no_grad():
        output = net(dummy_input)
    
    print(f"Output shape: {output.shape}")

if __name__ == "__main__":
    main()
