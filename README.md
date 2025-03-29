# Egg Analyzer 🥚  

Streamlit-приложение для анализа изображений яиц с использованием YOLO-модели. Определяет количество светлых и темных яиц на основе заданного порога яркости.  

[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app/)  

![Пример работы](screenshots/demo.gif)  

## Особенности  

- 🖼️ Загрузка изображений форматов JPG/PNG/JPEG  
- 🔍 Детекция яиц с помощью YOLO-модели  
- 📊 Классификация на светлые/темные по порогу яркости  
- ⚙️ Настройка параметров через интерфейс  
- 📍 Визуализация результатов с контурами яиц
- 📈 Отображение статистики в реальном времени  

## Установка и запуск  

1. Клонируйте репозиторий:
```bash 
git clone https://github.com/solidbell28/egg_segmentation.git 
cd egg-analyzer
```
2. Установите зависимости:
```bash
pip install -r requirements.txt
```
3. Запустите приложение
```bash
streamlit run app.py  
```

## Обучение модели
Весь процесс обучения и эксперименты доступны в Jupyter-ноутбуке:
🔬 [IPR Task1 Experiments на Kaggle](https://www.kaggle.com/code/kinging/ipr-task1-experiments)

Ноутбук содержит:
- Загрузку данных
- Обучение YOLO-модели
- Визуализацию результатов
- Конструирование API

## Данные
Исходный датасет для обучения доступен на Яндекс.Диске:
📦 [Ссылка на датасет](https://disk.yandex.ru/client/disk/IPR/Task_1)

Архив содержит:
- Исходные изображения яиц
- Разметку в формате YOLO
- Дополнительные метаданные
