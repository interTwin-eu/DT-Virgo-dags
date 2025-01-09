# syntax=docker/dockerfile:1
FROM romanoa77/pytools:0.1 AS compp



RUN python3 -m venv /opt/venv

ENV PATH=/opt/venv/bin:$PATH
COPY requirements.txt .

RUN  pip install --no-cache-dir --upgrade pip 


RUN --mount=type=cache,target=/root/.cache/pip pip install \
--no-cache-dir \
--compile \
--global-option="-Os" \
--global-option="-gO" \
--global-option="-Wl,--strip-all" \
-r requirements.txt

CMD [ "sleep", "infinity" ]
