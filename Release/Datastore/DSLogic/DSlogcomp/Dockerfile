# syntax=docker/dockerfile:1
FROM python:3.8.10-alpine AS comp

RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

FROM python:3.8.10-alpine

### Env var and cmd line arguments####
ARG srvd=/var/www

ENV ENDP_STAT=stat
ENV ENDP_DUMP=dumpLogF
ENV ENDP_DUMPF=dumpF
ENV ENDP_SEND=sendF
ENV DB_BURL=http://databuff-0.databuff.glitchflow.svc.cluster.local:8080/
ENV ENDP_DESC=dstat
ENV ENDP_UPD_DESC=upddsc
ENV ENDP_CLEAN=cleanall
ENV MAX_SIZE=2000




ENV GREET="Hello from Gunicorn server!"
ENV PATH=/opt/venv/bin:$PATH
######################################

RUN mkdir ${srvd}

COPY --from=comp /opt/venv /opt/venv
COPY  gunicorn.conf.py /etc
COPY app ${srvd}/app

WORKDIR ${srvd}



CMD [ "gunicorn", "-c","/etc/gunicorn.conf.py", "app:create_app('default')" ]
