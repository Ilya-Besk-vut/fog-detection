import os
from pathlib import Path
import cv2
import gradio as gr
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

from src.preprocessing.dcp_pipeline import compute_dark_channel_fast

IMAGE_SIZE = (32, 32)
CLASS_NAMES = ['Clear', 'Cloudy', 'Fog']
LOADED_MODELS = {}
SAVED_MODELS_DIR = Path('saved_models')


def scan_saved_models():
  models_dict = {}
  if SAVED_MODELS_DIR.exists():
    # Ищем файлы .keras и .h5
    for model_path in sorted(SAVED_MODELS_DIR.glob('*.keras')):
      models_dict[model_path.stem] = str(model_path)
    for model_path in sorted(SAVED_MODELS_DIR.glob('*.h5')):
      models_dict[model_path.stem] = str(model_path)

  # Дополнительно проверяем старый файл снаружи, если он существует
  external_baseline = Path('../best_model.keras')
  if external_baseline.exists():
    models_dict['Old Baseline (external)'] = str(external_baseline)

  return models_dict


MODELS_DICT = scan_saved_models()


def get_model(model_name):
  model_path = MODELS_DICT[model_name]
  if model_path not in LOADED_MODELS:
    print(f'Loading model from: {model_path}...')
    LOADED_MODELS[model_path] = load_model(model_path)
  return LOADED_MODELS[model_path]


def prepare_image_for_model(image_rgb: np.ndarray, model, img_size=(32, 32)):

  # 1. Защита от RGBA (убираем прозрачность, если загрузили PNG с альфа-каналом)
  if image_rgb.shape[-1] == 4:
    image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGBA2RGB)

  # 2. Gradio передает RGB -> переводим в BGR (формат OpenCV, на котором училась сеть)
  image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

  # 3. Узнаем, сколько каналов ждет загруженная модель: shape вида (None, 32, 32, C)
  expected_channels = model.input_shape[-1]

  # 4. Базовый ресайз и нормализация
  img_resized = cv2.resize(image_bgr, img_size).astype('float32') / 255.0

  if expected_channels == 4:
    # Рассчитываем темный канал на исходном разрешении и сжимаем
    dcp = compute_dark_channel_fast(image_bgr)
    dcp_resized = cv2.resize(dcp, img_size).astype('float32') / 255.0

    # Склеиваем 4 канала: [32, 32, 3] + [32, 32, 1] -> [32, 32, 4]
    features = np.dstack([img_resized, dcp_resized])
  else:
    features = img_resized

  # Добавляем размерность батча: (1, 32, 32, C)
  return np.expand_dims(features, axis=0)


def predict_fog(image, model_name):
  if image is None or not model_name:
    return None

  # 1. Загружаем выбранную модель
  model = get_model(model_name)

  # 2. Вызываем адаптивную подготовку данных под архитектуру модели
  data = prepare_image_for_model(image, model, img_size=IMAGE_SIZE)

  # 3. Инференс
  predictions = model.predict(data)[0]

  result = {}
  for i in range(len(CLASS_NAMES)):
    result[CLASS_NAMES[i]] = float(predictions[i])
  return result


def shutdown():
  print('\n[INFO] Working was terminated by user')
  os._exit(0)


# --- Gradio UI ---
default_choice = list(MODELS_DICT.keys())[0] if MODELS_DICT else None

with gr.Blocks(title='The fog detection system') as demo:
  gr.Markdown('# The fog detection system')
  gr.Markdown('Please upload an image and select a model for analysis')

  with gr.Row():
    with gr.Column():
      input_image = gr.Image(type='numpy', label='Upload image')
      model_dropdown = gr.Dropdown(
          choices=list(MODELS_DICT.keys()),
          value=default_choice,
          label='Select a model',
      )
      btn_predict = gr.Button('Analyze', variant='primary')

    with gr.Column():
      output_label = gr.Label(num_top_classes=3, label='Analysis result')

  gr.Markdown('---')
  btn_stop = gr.Button('Terminate the application', variant='stop')

  btn_predict.click(
      fn=predict_fog, inputs=[input_image, model_dropdown], outputs=output_label
  )

  btn_stop.click(fn=shutdown, inputs=None, outputs=None)


if __name__ == '__main__':
  demo.launch()