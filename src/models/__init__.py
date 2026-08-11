# src/models/__init__.py

from src.models.model_conf import build_model as build_custom_cnn
# from src.models.resnet import build_model as build_resnet


def get_model(model_name: str, input_shape=(32, 32, 3), num_classes=3):
    """Реестр всех твоих моделей из папки models/"""
    models_registry = {
        'custom_cnn': build_custom_cnn,
        # 'resnet': build_resnet,    # resnet.py
    }

    if model_name not in models_registry:
        raise ValueError(
            f"Модель '{model_name}' не найдена! "
            f"Доступные модели в реестре: {list(models_registry.keys())}"
        )

    return models_registry[model_name](
        input_shape=input_shape, num_classes=num_classes
    )