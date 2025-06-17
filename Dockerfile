FROM python:3.9-slim-buster

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

RUN python -m spacy download es_core_news_sm
RUN python -c "import nltk; nltk.download('stopwords', quiet=True)"

COPY . .

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "historyline_app.py", "--server.port=8501", "--server.address=0.0.0.0"]