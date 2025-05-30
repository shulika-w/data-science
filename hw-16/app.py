import streamlit as st

# Налаштування сторінки на широкий режим
st.set_page_config(
    page_title="Fashion Image Classification",
    page_icon="👗",
    layout="wide"
)
import gdown
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import json

# Словник перекладів
translations = {
    'en': {
        'page_title': 'Fashion Image Classification using CNN and VGG16',
        'select_model': 'Select Model',
        'about_app': 'About Application',
        'app_description': '''
This application allows you to classify fashion images using:
- Custom Convolutional Neural Network (CNN)
- Modified VGG16 Architecture
        ''',
        'classes': 'Model classifies images into 10 classes:',
        'upload_image': 'Upload an image',
        'uploaded_image': 'Uploaded image',
        'predicted_class': 'Predicted class',        
        'probability_distribution': 'Class Probability Distribution',
        'probability': 'Probability',
        'training_graphs': 'Training Graphs',
        'model_accuracy': 'Model Accuracy',
        'model_loss': 'Model Loss',
        'accuracy': 'Accuracy',
        'loss': 'Loss',
        'epoch': 'Epoch',
        'training': 'Training',
        'validation': 'Validation',
        'model_error': 'Model loading error. Please check model files.',
        'processing_error': 'Error processing image: ',
        'history_unavailable': 'Training history for model {} is unavailable'
    },
    'uk': {
        'page_title': 'Класифікація зображень одягу за допомогою CNN та VGG16',
        'select_model': 'Виберіть модель',
        'about_app': 'Про застосунок',
        'app_description': '''
Цей застосунок дозволяє класифікувати зображення одягу за допомогою:
- Власної згорткової нейронної мережі (CNN)
- Модифікованої архітектури VGG16
        ''',
        'classes': 'Модель класифікує зображення на 10 класів:',
        'upload_image': 'Завантажте зображення',
        'uploaded_image': 'Завантажене зображення',
        'predicted_class': 'Передбачений клас',        
        'probability_distribution': 'Розподіл ймовірностей по класах',
        'probability': 'Ймовірність',
        'training_graphs': 'Графіки тренування',
        'model_accuracy': 'Точність моделі',
        'model_loss': 'Втрати моделі',
        'accuracy': 'Точність',
        'loss': 'Втрати',
        'epoch': 'Епоха',
        'training': 'Тренування',
        'validation': 'Валідація',
        'model_error': 'Помилка завантаження моделі. Перевірте наявність файлів моделей.',
        'processing_error': 'Помилка при обробці зображення: ',
        'history_unavailable': 'Історія тренування для моделі {} недоступна'
    }
}

# Вибір мови у сайдбарі
language = st.sidebar.selectbox('🌐 Language / Мова', ['en', 'uk'], format_func=lambda x: 'English' if x == 'en' else 'Українська')
t = translations[language]

# Функція для відображення графіків
def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Графік точності
    ax1.plot(history['accuracy'])
    ax1.plot(history['val_accuracy'])
    ax1.set_title(t['model_accuracy'])
    ax1.set_ylabel(t['accuracy'])
    ax1.set_xlabel(t['epoch'])
    ax1.legend([t['training'], t['validation']], loc='lower right')
    
    # Графік втрат
    ax2.plot(history['loss'])
    ax2.plot(history['val_loss'])
    ax2.set_title(t['model_loss'])
    ax2.set_ylabel(t['loss'])
    ax2.set_xlabel(t['epoch'])
    ax2.legend([t['training'], t['validation']], loc='upper right')
    
    return fig

# Завантаження моделей
@st.cache_resource
def load_cnn_model():
    try:
        model = load_model('cnn_model.keras', compile=False)        
        return model
    except Exception as e:
        st.error(f"Помилка завантаження CNN моделі: {str(e)}")
        return None

@st.cache_resource
def load_vgg16_model():
    model_path = "vVGG16_model.keras"
    file_id = "1wF6T2V1_EmjwmPJ4sfeEVWm6ggHQltMw"
    url = f"https://drive.google.com/uc?id={file_id}"
    
    try:
        # Завантажуємо файл, якщо він ще не існує
        if not os.path.exists(model_path):
            with st.spinner("Завантаження моделі..."):
                gdown.download(url, model_path, quiet=False)

        # Завантажуємо модель
        model = load_model(model_path, compile=False)
        return model

    except Exception as e:
        st.error(f"Помилка завантаження VGG16 моделі: {str(e)}")
        return None

# Функції для підготовки зображень
def preprocess_image_for_cnn(img):
    img = img.resize((28, 28))
    img = img.convert('L')  
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def preprocess_image_for_vgg16(img):
    img = img.resize((32, 32))
    img = img.convert('RGB')  
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Інтерфейс Streamlit
st.markdown(f"# :rainbow[{t['page_title']}]")

# Клас-назви для Fashion MNIST
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Вибір моделі
model_option = st.sidebar.selectbox(t['select_model'], ['CNN', 'VGG16'])

# Додаткова інформація
st.sidebar.markdown("---")
st.sidebar.write(f"### {t['about_app']}")
st.sidebar.write(t['app_description'])
st.sidebar.write(t['classes'])
st.sidebar.write("\n".join([f"- {name}" for name in class_names]))

# Завантаження моделі
if model_option == 'CNN':
    model = load_cnn_model()
else:
    model = load_vgg16_model()

# Завантаження зображення
uploaded_file = st.file_uploader(t['upload_image'], type=["png", "jpg", "jpeg"])

if uploaded_file is not None and model is not None:
    try:
        img = image.load_img(uploaded_file)
        st.image(img, caption=t['uploaded_image'], width=400)

        if model_option == 'CNN':
            img_array = preprocess_image_for_cnn(img)
        else:
            img_array = preprocess_image_for_vgg16(img)

        # Передбачення
        prediction = model.predict(img_array)       
        
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100
        
        st.success(f"**{t['predicted_class']}:** {predicted_class} ({t['probability']}: {confidence:.2f}%)")
        
        col1, col2 = st.columns([2, 1]) 
        with col1:            
            plt.figure(figsize=(12, 6))  
            fig, ax = plt.subplots()
            y_pos = np.arange(len(class_names))
            bars = ax.barh(y_pos, prediction[0] * 100)
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(class_names, fontsize=10)  
            ax.invert_yaxis()
            ax.set_xlabel(t['probability'] + ' (%)', fontsize=10)  
            ax.set_title(t['probability_distribution'], fontsize=12, pad=15)  
            
            ax.tick_params(axis='x', labelsize=10)
            ax.grid(True, linestyle='--', alpha=0.7)
            
            for i, v in enumerate(prediction[0] * 100):
                ax.text(v + 1, i, f'{v:.1f}%', va='center', fontsize=9)
            
            plt.tight_layout()
            st.pyplot(fig)
        
    except Exception as e:
        st.error(f"{t['processing_error']}{str(e)}")

    # Відображення історії тренування
    try:
        history_file = 'cnn_training_history.json' if model_option == 'CNN' else 'VGG16_training_history.json'
        with open(history_file, 'r') as f:
            history = json.load(f)
            st.write(f"### {t['training_graphs']}:")
            st.pyplot(plot_training_history(history))
    except Exception as e:
        st.info(t['history_unavailable'].format(model_option))

elif model is None:
    st.error(t['model_error'])