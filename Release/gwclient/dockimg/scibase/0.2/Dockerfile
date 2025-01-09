# syntax=docker/dockerfile:1
FROM romanoa77/scibase:0.1 AS compp1

ENV PATH=/opt/venv/bin:$PATH

RUN  apt-get install -y  libopenblas-dev liblapack-dev pkg-config  python3-dev

COPY requirements2.txt .

 

RUN --mount=type=cache,target=/root/.cache/pip pip install \
--no-cache-dir \
--compile \
--global-option="-Os" \
--global-option="-gO" \
--global-option="-Wl,--strip-all" \
-r requirements2.txt



COPY requirements1.txt .

 

RUN --mount=type=cache,target=/root/.cache/pip pip install \
--no-cache-dir \
--compile \
--global-option="-Os" \
--global-option="-gO" \
--global-option="-Wl,--strip-all" \
-r requirements1.txt







CMD [ "sleep", "infinity" ]
