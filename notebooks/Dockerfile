FROM python:3.11

WORKDIR /notebooks

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir gdown

RUN apt-get update && apt-get install -y supervisor unzip

COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

RUN gdown "https://drive.google.com/uc?id=18NdKeC-kZ3OP5Qpk_xmkFc4NUQKVH1z1"
RUN unzip data.zip -d /notebooks
RUN rm -f data.zip

COPY . .

CMD ["/usr/bin/supervisord"]
