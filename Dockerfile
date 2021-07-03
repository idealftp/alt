FROM python:3.6

ENV DEBIAN_FRONTEND noninteractive

ADD . /root/

WORKDIR /root/

RUN apt-get update -y && apt-get install ffmpeg libsm6 libxext6 python3-pip -y && \
ln -sf /usr/bin/python3.6 /usr/bin/python && \
python -m pip install -r requirements.txt

EXPOSE 3000

CMD python flastest.py