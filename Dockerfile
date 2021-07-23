FROM python:3.8

WORKDIR /mediapipe-paint

COPY requirements.txt .

RUN pip install -r requirements.txt
RUN apt-get -qq update \
    && apt-get -y install \
	  ffmpeg \
	  libsm6 \
	  libxext6 \
	&& rm -rf /var/lib/apt/lists/*

COPY paint_to_webcam.py painting_utils.py ./app/