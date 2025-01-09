# syntax=docker/dockerfile:1
FROM romanoa77/gwpyimg:0.1 AS build

ENV PATH=/opt/venv/bin:$PATH
ENV PATH="/root/.cargo/bin:$PATH"




RUN apt-get install -y libhdf5-dev
RUN curl --proto '=https' --tlsv1.2 -sSf  https://sh.rustup.rs | sh -s -- -y
RUN rustup update






COPY requirements5.txt .

 

RUN --mount=type=cache,target=/root/.cache/pip pip install \
--no-cache-dir \
--compile \
--global-option="-Os" \
--global-option="-gO" \
--global-option="-Wl,--strip-all" \
-r requirements5.txt

FROM python:3.10-slim

ENV PATH=/opt/venv/bin:$PATH
COPY --from=build /opt/venv /opt/venv
COPY --from=build /usr/lib/x86_64-linux-gnu/ /usr/lib/x86_64-linux-gnu/



CMD [ "sleep", "infinity" ]
