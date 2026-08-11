# train.py
import argparse
import os
import random
import numpy as np
import tensorflow as tf
import yaml

from src.models import get_model
from src.preprocessing import load_data


# function for fixing random state
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def main(config_path: str):
    # Loading configurqtion from yaml 
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    print(f"Strart {config['experiment_name']}")

    # fix random state (from yaml)
    seed = config['data'].get('random_state', 42)
    set_seed(seed)

    #  reading .npz file
    X_train, X_val, y_train, y_val = load_data(
        npz_path=config['data']['dataset_path'],
        val_split=config['data']['val_split'],
        random_state=seed,
    )

    #  model initialization
    img_size = config['data']['img_size']
    input_shape = (img_size[0], img_size[1], 3)
    num_classes = config['model']['num_classes']

    model = get_model(
        model_name=config['model']['name'],
        input_shape=input_shape,
        num_classes=num_classes,
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config['training']['learning_rate']
    )
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    # Training
    print('Training')
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=config['training']['epochs'],
        batch_size=config['data']['batch_size'],
    )

    # saving the trainded model 
    save_path = config['training']['save_path']
    
    model.save(save_path)
    print(f"Model saved '{save_path}'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/baseline.yaml')
    args = parser.parse_args()

    main(args.config)