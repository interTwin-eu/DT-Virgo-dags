# syntax=docker/dockerfile:1
FROM romanoa77/scibase:0.2 AS compp2

ENV PATH=/opt/venv/bin:$PATH



COPY requirements3.txt .

 

RUN --mount=type=cache,target=/root/.cache/pip pip install \
--no-cache-dir \
--compile \
--global-option="-Os" \
--global-option="-gO" \
--global-option="-Wl,--strip-all" \
-r requirements3.txt



CMD [ "sleep", "infinity" ]
