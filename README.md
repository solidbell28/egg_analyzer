# Egg Analyzer 🥚  

Streamlit-приложение для анализа изображений яиц с использованием YOLO-модели. Определяет количество светлых и темных яиц на основе заданного порога яркости.  

[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app/)  

![Пример работы](screenshots/demo.gif)  

## Особенности  

- 🖼️ Загрузка изображений форматов JPG/PNG/JPEG  
- 🔍 Детекция яиц с помощью YOLO-модели  
- 📊 Классификация на светлые/темные по порогу яркости  
- ⚙️ Настройка параметров через интерфейс  
- 📍 Визуализация результатов с bounding boxes  
- 📈 Отображение статистики в реальном времени  

## Установка и запуск  

1. Клонируйте репозиторий:
```
bash 
git clone https://github.com/yourusername/egg-analyzer.git  
cd egg-analyzer
```
2. Установите зависимости:
```
pip install -r requirements.txt
```
3. Запустите приложение
```
streamlit run app.py  
```
