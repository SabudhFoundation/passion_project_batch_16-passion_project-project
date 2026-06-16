"""
Grad-CAM Explainability using
Pretrained InceptionV3 (ImageNet)

This example demonstrates:
1. Loading pretrained InceptionV3
2. Running image classification
3. Generating Grad-CAM heatmap
4. Overlaying attention map on image

Install:
pip install torch torchvision pillow matplotlib opencv-python numpy
"""

import torch
import torch.nn.functional as F
from torchvision import models
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

# ---------------------------------------------------------
# 1. Load Pretrained InceptionV3
# ---------------------------------------------------------

weights = models.Inception_V3_Weights.IMAGENET1K_V1

model = models.inception_v3(weights=weights)
model.eval()

print("Loaded pretrained InceptionV3")

# ---------------------------------------------------------
# 2. Load Image
# ---------------------------------------------------------

image_path = "test1.jpg"

image = Image.open(image_path).convert("RGB")

# Keep original image for visualization
original_image = np.array(image)

# ---------------------------------------------------------
# 3. Preprocessing
# ---------------------------------------------------------

preprocess = weights.transforms()

input_tensor = preprocess(image).unsqueeze(0)

# ---------------------------------------------------------
# 4. Hook Feature Maps and Gradients
# ---------------------------------------------------------

activations = None
gradients = None

# Target layer for GradCAM
target_layer = model.Mixed_7c

# Forward hook
def forward_hook(module, input, output):
    global activations
    activations = output.detach()

# Backward hook
def backward_hook(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0].detach()

# Register hooks
forward_handle = target_layer.register_forward_hook(forward_hook)
backward_handle = target_layer.register_full_backward_hook(backward_hook)

# ---------------------------------------------------------
# 5. Forward Pass
# ---------------------------------------------------------

output = model(input_tensor)

# Prediction probabilities
probabilities = F.softmax(output[0], dim=0)

# Predicted class
predicted_class = torch.argmax(probabilities).item()

categories = weights.meta["categories"]

print("\nPrediction:")
print(categories[predicted_class])

# ---------------------------------------------------------
# 6. Backward Pass
# ---------------------------------------------------------

model.zero_grad()

# Target score
target = output[0][predicted_class]

# Compute gradients
target.backward()

# ---------------------------------------------------------
# 7. Compute GradCAM Heatmap
# ---------------------------------------------------------

# Global average pooling on gradients
pooled_gradients = torch.mean(
    gradients,
    dim=[0, 2, 3]
)

# Weight feature maps
for i in range(activations.shape[1]):
    activations[:, i, :, :] *= pooled_gradients[i]

# Average across channels
heatmap = torch.mean(activations, dim=1).squeeze()

# Apply ReLU
heatmap = F.relu(heatmap)

# Normalize
heatmap /= torch.max(heatmap)

# Convert to numpy
heatmap = heatmap.cpu().numpy()

# ---------------------------------------------------------
# 8. Resize Heatmap
# ---------------------------------------------------------

heatmap = cv2.resize(
    heatmap,
    (original_image.shape[1], original_image.shape[0])
)

# Convert to 0-255
heatmap_uint8 = np.uint8(255 * heatmap)

# Apply colormap
heatmap_color = cv2.applyColorMap(
    heatmap_uint8,
    cv2.COLORMAP_JET
)

# ---------------------------------------------------------
# 9. Overlay Heatmap on Original Image
# ---------------------------------------------------------

overlay = cv2.addWeighted(
    original_image,
    0.6,
    heatmap_color,
    0.4,
    0
)

# ---------------------------------------------------------
# 10. Visualization
# ---------------------------------------------------------

plt.figure(figsize=(18, 6))

# Original image
plt.subplot(1, 3, 1)
plt.imshow(original_image)
plt.title("Original Image")
plt.axis("off")

# Heatmap
plt.subplot(1, 3, 2)
plt.imshow(heatmap, cmap="jet")
plt.title("GradCAM Heatmap")
plt.axis("off")

# Overlay
plt.subplot(1, 3, 3)
plt.imshow(overlay)
plt.title(
    f"Prediction: {categories[predicted_class]}"
)
plt.axis("off")

plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 11. Cleanup Hooks
# ---------------------------------------------------------

forward_handle.remove()
backward_handle.remove()