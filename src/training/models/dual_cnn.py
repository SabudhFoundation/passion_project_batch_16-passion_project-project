import os
import gc
import torch

from torch import nn, optim


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# Spatial Stream

def build_spatial_stream():

    return nn.Sequential(

        nn.Conv2d(3, 32, 3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),

        nn.Conv2d(32, 32, 3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),

        nn.MaxPool2d(2),

        nn.Conv2d(32, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),

        nn.Conv2d(64, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),

        nn.MaxPool2d(2),

        nn.Conv2d(64, 128, 3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),

        nn.Conv2d(128, 128, 3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),

        nn.MaxPool2d(2),

        nn.Conv2d(128, 256, 3, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),

        nn.AdaptiveAvgPool2d((4, 4)),

        nn.Flatten()
    )


# Frequency Stream

def build_frequency_stream():

    return nn.Sequential(

        nn.BatchNorm2d(3),

        nn.Conv2d(3, 32, 3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),

        nn.Conv2d(32, 32, 3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),

        nn.MaxPool2d(2),

        nn.Conv2d(32, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),

        nn.Conv2d(64, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),

        nn.MaxPool2d(2),

        nn.Conv2d(64, 128, 3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),

        nn.Conv2d(128, 128, 3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),

        nn.MaxPool2d(2),

        nn.Conv2d(128, 256, 3, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),

        nn.AdaptiveAvgPool2d((4, 4)),

        nn.Flatten()
    )


# Dual Stream CNN

class DualStreamCNN(nn.Module):

    def __init__(self):

        super(DualStreamCNN, self).__init__()

        self.spatial_net = build_spatial_stream()

        self.freq_net = build_frequency_stream()

        self.spatial_dim = 256 * 4 * 4

        self.freq_dim = 256 * 4 * 4

        self.classifier = nn.Sequential(

            nn.Linear(
                self.spatial_dim + self.freq_dim,
                1024
            ),

            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(1024, 256),

            nn.ReLU(),

            nn.Dropout(0.4),

            nn.Linear(256, 64),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(64, 1)
        )

    def forward(self, x):

        # Spatial Features

        spatial_feat = self.spatial_net(x)

        # FFT Features

        x_fft = torch.fft.fftn(
            x,
            dim=(-2, -1)
        )

        x_fft = torch.fft.fftshift(x_fft)

        x_fft_mag = torch.log(
            torch.abs(x_fft) + 1e-8
        )

        freq_feat = self.freq_net(x_fft_mag)

        # Feature Fusion

        combined = torch.cat(
            (spatial_feat, freq_feat),
            dim=1
        )

        output = self.classifier(combined)

        return output


# Loss Function

def get_loss_function():

    return nn.BCEWithLogitsLoss()


# Optimizer

def get_optimizer(
    model,
    learning_rate
):

    return optim.Adam(
        model.parameters(),
        lr=learning_rate
    )


# Validation

def validate_model(
    model,
    valid_loader
):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in valid_loader:

            images = images.to(device)

            labels = labels.to(device)

            outputs = model(images)

            preds = (
                torch.sigmoid(outputs) > 0.5
            ).int().squeeze()

            correct += (
                preds == labels
            ).sum().item()

            total += labels.size(0)

            del images
            del labels
            del outputs
            del preds

            torch.cuda.empty_cache()

    return correct / total


# Test

def test_model(
    model,
    test_loader
):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)

            labels = labels.to(device)

            outputs = model(images)

            preds = (
                torch.sigmoid(outputs) > 0.5
            ).int().squeeze()

            correct += (
                preds == labels
            ).sum().item()

            total += labels.size(0)

            del images
            del labels
            del outputs
            del preds

            torch.cuda.empty_cache()

    return correct / total


# Training

def train_model(
    model,
    train_loader,
    valid_loader,
    criterion,
    optimizer,
    epochs,
    model_path
):

    best_accuracy = 0

    for epoch in range(epochs):

        model.train()

        total_loss = 0

        for batch_idx, (images, labels) in enumerate(train_loader):

            images = images.to(device)

            labels = labels.float().unsqueeze(1).to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 20 == 0:

                print(

                    f"Epoch [{epoch+1}/{epochs}] "

                    f"Batch [{batch_idx+1}/{len(train_loader)}] "

                    f"Loss: {loss.item():.4f}"
                )

            del images
            del labels
            del outputs
            del loss

            torch.cuda.empty_cache()

        accuracy = validate_model(
            model,
            valid_loader
        )

        print(
            f"\nEpoch [{epoch+1}/{epochs}] "
            f"Validation Accuracy: {accuracy:.4f}\n"
        )

        if accuracy > best_accuracy:

            best_accuracy = accuracy

            torch.save(
                model.state_dict(),
                model_path
            )

            print("\nBest Model Saved\n")

        gc.collect()

        torch.cuda.empty_cache()

    return best_accuracy


# Pipeline

def train_dual_cnn_pipeline(
    train_loader,
    valid_loader,
    test_loader,
    model_dir,
    epochs,
    learning_rate
):

    model = DualStreamCNN().to(device)

    criterion = get_loss_function()

    optimizer = get_optimizer(
        model,
        learning_rate
    )

    os.makedirs(
        model_dir,
        exist_ok=True
    )

    model_path = os.path.join(
        model_dir,
        "dual_stream_cnn.pth"
    )

    best_accuracy = train_model(

        model,
        train_loader,
        valid_loader,
        criterion,
        optimizer,
        epochs,
        model_path
    )

    print(
        f"\nBest Validation Accuracy: "
        f"{best_accuracy:.4f}"
    )

    model.load_state_dict(
        torch.load(model_path)
    )

    test_accuracy = test_model(
        model,
        test_loader
    )

    print(
        f"\nTest Accuracy: "
        f"{test_accuracy:.4f}"
    )