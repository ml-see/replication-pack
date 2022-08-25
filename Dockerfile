# syntax=docker/dockerfile:1
FROM python:3.8.12

COPY app/requirements.txt requirements.txt
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
RUN pip3 install --no-cache-dir sentencepiece
RUN pip3 install -r requirements.txt
RUN python -m nltk.downloader stopwords

COPY . .

RUN apt-get -y update
RUN mkdir -p /usr/share/man/man1/
RUN apt-get install -y openjdk-11-jdk
RUN apt-get install -y openjdk-11-jre
RUN update-alternatives --config java
RUN update-alternatives --config javac

CMD ["python", "app/ml_predictions.py"]