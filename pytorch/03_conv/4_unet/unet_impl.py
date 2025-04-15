import torch
from torch import nn, functional
from matplotlib import pyplot

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        
        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)
        
        # Expansive Path (Decoder)
        self.decoder4 = self.conv_block(1024 + 512, 512)
        self.decoder3 = self.conv_block(512 + 256, 256)
        self.decoder2 = self.conv_block(256 + 128, 128)
        self.decoder1 = self.conv_block(128 + 64, 64)
        
        # Final Output Layer
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def conv_block(self, in_channels, out_channels):
        # Helper function to create a block of two convolution layers with ReLU activations.
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        return block
    
    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(functional.F.max_pool2d(enc1, 2))
        enc3 = self.encoder3(functional.F.max_pool2d(enc2, 2))
        enc4 = self.encoder4(functional.F.max_pool2d(enc3, 2))

        # Bottleneck
        bottleneck = self.bottleneck(functional.F.max_pool2d(enc4, 2))

        # Decoder with Skip Connections
        dec4 = self.decoder4(torch.cat([functional.F.interpolate(bottleneck, scale_factor=2, mode='bilinear', align_corners=True), enc4], 1))
        dec3 = self.decoder3(torch.cat([functional.F.interpolate(dec4, scale_factor=2, mode='bilinear', align_corners=True), enc3], 1))
        dec2 = self.decoder2(torch.cat([functional.F.interpolate(dec3, scale_factor=2, mode='bilinear', align_corners=True), enc2], 1))
        dec1 = self.decoder1(torch.cat([functional.F.interpolate(dec2, scale_factor=2, mode='bilinear', align_corners=True), enc1], 1))

        # Final Output
        out = self.final_conv(dec1)

        return out
    
# 模拟数据集：假设每个图像大小为 256x256，输入图像有3个通道（RGB），标签为单通道（背景和目标）
def generate_random_data(batch_size=4, height=256, width=256):
    # 随机生成 RGB 图像，值范围 [0, 1]
    images = torch.randn(batch_size, 3, height, width)
    
    # 随机生成标签（0或1），代表背景（0）和目标（1）
    labels = torch.randint(0, 2, (batch_size, 1, height, width), dtype=torch.float32)
    
    return images, labels

if __name__ == "__main__":
    images, labels = generate_random_data(batch_size=4)
    
    pyplot.figure(figsize=(12, 6))
    pyplot.subplot(1, 2, 1)
    pyplot.imshow(images[0].permute(1, 2, 0).numpy())
    pyplot.title('Input Image')
    pyplot.subplot(1, 2, 2)
    pyplot.imshow(labels[0].squeeze().numpy(), cmap='gray')
    pyplot.title('Ground Truth Label')
    pyplot.show()
    
    # Define the model
    model = UNet(in_channels=3, out_channels=1)  # Example: 3 input channels (RGB image), 1 output channel (segmentation mask)
    
    # Create a random input tensor (Batch of 1 image with 3 channels, size 256x256)
    input_image = torch.randn(1, 3, 256, 256)  # (batch_size, channels, height, width)
    
    # Get the model's output
    output = model(input_image)
    
    # Print the output shape (should be same height and width, with output channels)
    print("Output shape:", output.shape)
    
    # Define a loss function (for binary segmentation)
    criterion = nn.BCEWithLogitsLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    epochs = 10
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        output = model(images)
        
        # Calculate loss
        loss = criterion(output, labels)  # target should be the ground truth segmentation map
        loss.backward()
        # Update weights
        optimizer.step()

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')