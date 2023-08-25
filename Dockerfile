FROM python:3.9

RUN apt-get update &&\
    apt-get install -y libproj-dev gdal-bin
    
COPY requirements.txt /opt/run/requirements.txt

RUN pip install -r /opt/run/requirements.txt

COPY main.py /opt/run/main.py
COPY ./src /opt/run/src
COPY ./assets/ /opt/run/assets
COPY .cdsapirc /root/.cdsapirc

WORKDIR /opt/run/

CMD python main.py