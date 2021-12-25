FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

RUN apt-get update && apt-get install -y libglib2.0-0  libsm6 libxext6 libxrender-dev

COPY . .

RUN pip install --no-cache-dir -r ./requirements.txt

EXPOSE 5000

CMD gunicorn --timeout 200 -w 3 -b 0.0.0.0:5000 drawing:app
