# syntax=docker/dockerfile:1
FROM python:3.8.10-slim-buster

ENV DS_STAT=bufstat
ENV DS_SEND=stream
ENV DS_FREEZE=freezeds
ENV DS_BURL=http://dslogic.glitchflow.svc.cluster.local:5000/
ENV DS_FLUSH=flushbuf
ENV DS_DESC=dsdesc
ENV DS_DUMPF=dumpF
ENV MAX_SIZE=2000


WORKDIR /code


COPY ./requirements.txt /code/requirements.txt

 
RUN --mount=type=cache,target=/root/.cache/pip pip install   -r /code/requirements.txt


COPY ./app /code/app


CMD ["fastapi", "run", "app/main.py", "--port", "8000"]