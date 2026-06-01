"""DataLoaders para MNIST, Fashion-MNIST y CIFAR-10.

Estrategia recomendada del briefing:
  - MNIST / Fashion-MNIST: para verificar correctitud (~1h de entrenamiento)
  - CIFAR-10: para el run principal y el titular de FID

No hay fuga de datos: train/val/test son splits disjuntos.
"""
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from typing import Tuple


def _normalize_to_minus_one_one() -> transforms.Normalize:
    """Mapea [0,1] a [-1, 1]; el modelo espera esta escala."""
    return transforms.Normalize(mean=[0.5], std=[0.5])


def _normalize_rgb_to_minus_one_one() -> transforms.Normalize:
    return transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])


def get_mnist_loaders(
    data_root: str = "./data/raw",
    batch_size: int = 128,
    num_workers: int = 4,
    val_fraction: float = 0.1,
    random_seed: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Retorna (train_loader, val_loader, test_loader) para MNIST.

    Las imagenes se escalan a [-1,1] y se convierten a tensor.
    """
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        _normalize_to_minus_one_one(),
    ])
    full_train_dataset = datasets.MNIST(data_root, train=True, download=True, transform=image_transform)
    test_dataset = datasets.MNIST(data_root, train=False, download=True, transform=image_transform)

    num_val = int(len(full_train_dataset) * val_fraction)
    num_train = len(full_train_dataset) - num_val
    train_dataset, val_dataset = random_split(
        full_train_dataset, [num_train, num_val],
        generator=torch.Generator().manual_seed(random_seed),
    )

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        drop_last=True,
    )
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader


def get_fashion_mnist_loaders(
    data_root: str = "./data/raw",
    batch_size: int = 128,
    num_workers: int = 4,
    val_fraction: float = 0.1,
    random_seed: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Retorna (train_loader, val_loader, test_loader) para Fashion-MNIST."""
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        _normalize_to_minus_one_one(),
    ])
    full_train_dataset = datasets.FashionMNIST(data_root, train=True, download=True, transform=image_transform)
    test_dataset = datasets.FashionMNIST(data_root, train=False, download=True, transform=image_transform)

    num_val = int(len(full_train_dataset) * val_fraction)
    num_train = len(full_train_dataset) - num_val
    train_dataset, val_dataset = random_split(
        full_train_dataset, [num_train, num_val],
        generator=torch.Generator().manual_seed(random_seed),
    )

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        drop_last=True,
    )
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader


def get_cifar10_loaders(
    data_root: str = "./data/raw",
    batch_size: int = 128,
    num_workers: int = 8,
    val_fraction: float = 0.1,
    random_seed: int = 0,
    use_horizontal_flip_augmentation: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Retorna (train_loader, val_loader, test_loader) para CIFAR-10.

    El paper usa flip horizontal como unica augmentacion.
    num_workers=8 es apropiado para 16 nucleos de CPU.
    """
    augmentations_for_train = [transforms.RandomHorizontalFlip()] if use_horizontal_flip_augmentation else []

    train_transform = transforms.Compose([
        *augmentations_for_train,
        transforms.ToTensor(),
        _normalize_rgb_to_minus_one_one(),
    ])
    val_test_transform = transforms.Compose([
        transforms.ToTensor(),
        _normalize_rgb_to_minus_one_one(),
    ])

    full_train_dataset = datasets.CIFAR10(data_root, train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(data_root, train=False, download=True, transform=val_test_transform)

    num_val = int(len(full_train_dataset) * val_fraction)
    num_train = len(full_train_dataset) - num_val
    train_dataset, val_dataset = random_split(
        full_train_dataset, [num_train, num_val],
        generator=torch.Generator().manual_seed(random_seed),
    )
    # val no usa augmentacion: reemplazar transform
    val_dataset.dataset = datasets.CIFAR10(data_root, train=True, download=False, transform=val_test_transform)

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        drop_last=True,
    )
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader
