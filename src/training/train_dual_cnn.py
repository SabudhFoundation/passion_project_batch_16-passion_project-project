import os

from torchvision import datasets, transforms

from torch.utils.data import DataLoader

from models.dual_cnn import train_dual_cnn_pipeline


def get_transforms(image_size):

    transform = transforms.Compose([

        transforms.Resize(
            (image_size, image_size)
        ),

        transforms.RandomHorizontalFlip(),

        transforms.RandomRotation(10),

        transforms.ToTensor(),

        transforms.Normalize(

            mean=[0.5, 0.5, 0.5],

            std=[0.5, 0.5, 0.5]
        )
    ])

    return transform


def load_datasets(
    dataset_dir,
    transform
):

    train_dataset = datasets.ImageFolder(

        root=os.path.join(
            dataset_dir,
            "train"
        ),

        transform=transform
    )

    valid_dataset = datasets.ImageFolder(

        root=os.path.join(
            dataset_dir,
            "valid"
        ),

        transform=transform
    )

    test_dataset = datasets.ImageFolder(

        root=os.path.join(
            dataset_dir,
            "test"
        ),

        transform=transform
    )

    print(
        f"\nClasses: "
        f"{train_dataset.class_to_idx}"
    )

    print(
        f"Train Images: "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation Images: "
        f"{len(valid_dataset)}"
    )

    print(
        f"Test Images: "
        f"{len(test_dataset)}"
    )

    return (
        train_dataset,
        valid_dataset,
        test_dataset
    )


def create_dataloaders(
    train_dataset,
    valid_dataset,
    test_dataset,
    batch_size
):

    train_loader = DataLoader(

        train_dataset,

        batch_size=batch_size,

        shuffle=True,

        num_workers=0
    )

    valid_loader = DataLoader(

        valid_dataset,

        batch_size=batch_size,

        shuffle=False,

        num_workers=0
    )

    test_loader = DataLoader(

        test_dataset,

        batch_size=batch_size,

        shuffle=False,

        num_workers=0
    )

    return (
        train_loader,
        valid_loader,
        test_loader
    )


def run_dual_cnn_training(
    dataset_dir,
    model_dir,
    epochs,
    batch_size,
    image_size,
    learning_rate
):

    print("\nLoading Dataset...\n")

    transform = get_transforms(
        image_size
    )

    (
        train_dataset,
        valid_dataset,
        test_dataset

    ) = load_datasets(

        dataset_dir,
        transform
    )

    print("\nCreating DataLoaders...\n")

    (
        train_loader,
        valid_loader,
        test_loader

    ) = create_dataloaders(

        train_dataset,
        valid_dataset,
        test_dataset,
        batch_size
    )

    print("\nStarting Dual Stream CNN Training...\n")

    train_dual_cnn_pipeline(

        train_loader=train_loader,

        valid_loader=valid_loader,

        test_loader=test_loader,

        model_dir=model_dir,

        epochs=epochs,

        learning_rate=learning_rate
    )

    print("\nDual Stream CNN Training Completed!\n")


if __name__ == "__main__":

    BASE_DIR = os.path.dirname(
        os.path.abspath(__file__)
    )

    dataset_dir = os.path.join(
        BASE_DIR,
        "../../real-vs-fake"
    )

    model_dir = os.path.join(
        BASE_DIR,
        "../../models"
    )

    run_dual_cnn_training(

        dataset_dir=dataset_dir,

        model_dir=model_dir,

        epochs=10,

        batch_size=8,

        image_size=192,

        learning_rate=0.0001
    )