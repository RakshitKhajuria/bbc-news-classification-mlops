FROM python:3.8-slim-buster
WORKDIR /bbc-news-classification-mlops
COPY . /bbc-news-classification-mlops
RUN apt update -y
RUN apt install awscli -y
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 unzip -y && pip install -r requirements.txt
COPY requirements.txt ./requirements.txt
RUN python -m nltk.downloader stopwords
RUN python -m nltk.downloader wordnet
RUN python -m nltk.downloader punkt
RUN python -m nltk.downloader averaged_perceptron_tagger
RUN python -m nltk.downloader omw-1.4
CMD ["python", "app_flask.py"]
