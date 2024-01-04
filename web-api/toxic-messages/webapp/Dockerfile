FROM python:3.10

RUN mkdir /face_detection_age_classification

COPY requirements.txt /toxic-messages/requirements.txt

COPY . /toxic-messages

WORKDIR /toxic-messages

RUN pip3 install --upgrade pip

RUN pip install  --prefer-binary --no-cache-dir --upgrade -r requirements.txt

EXPOSE 12345

CMD ["flask", "run", "--host", "0.0.0.0", "--port", "12345"]