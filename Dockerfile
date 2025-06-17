# Usa una imagen base de Python con Debian, estable y compatible con Spacy
FROM python:3.9-slim-buster

# Establece el directorio de trabajo
WORKDIR /app

# Copia el archivo de requisitos e instala las dependencias de Python
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# INSTALACIÓN Y DESCARGA DE MODELOS
# Descarga el modelo de SpaCy
RUN python -m spacy download es_core_news_sm
# Descarga las stopwords de NLTK
RUN python -c "import nltk; nltk.download('stopwords', quiet=True)"

# NUEVO: Precargar SentenceTransformer model durante la construcción
# Esto descargará los archivos del modelo en la imagen Docker
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# NUEVO: Precargar SpaCy model para asegurar que esté listo y no tarde en el arranque
# Esto no es para el spacy.load en sí, sino para asegurar que el path es correcto y la primera carga sea rápida
# El modelo ya está descargado por el paso anterior de spacy download

# Copia el resto de tu código de aplicación al contenedor
COPY . .

# Expone el puerto por defecto de Streamlit
EXPOSE 8501

# Comando para ejecutar la aplicación Streamlit
ENTRYPOINT ["streamlit", "run", "historyline_app.py", "--server.port=8501", "--server.address=0.0.0.0"]