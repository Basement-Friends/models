FROM python:3.10

RUN mkdir /face_detection_age_classification

RUN mkdir /face_detection_age_classification/temp

COPY requirements.txt /face_detection_age_classification/requirements.txt

COPY . /face_detection_age_classification

WORKDIR /face_detection_age_classification

RUN pip3 install --upgrade pip

RUN pip install  --prefer-binary --no-cache-dir --upgrade -r requirements.txt --timeout 360

RUN apt-get update
RUN apt install -y libgl1-mesa-glx

EXPOSE 23456

CMD ["flask", "run", "--host", "0.0.0.0", "--port", "23456"]