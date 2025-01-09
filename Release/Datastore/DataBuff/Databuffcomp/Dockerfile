# syntax=docker/dockerfile:1
FROM golang:1.22.4-alpine AS build

WORKDIR /app/class
COPY class .

WORKDIR /app/main
COPY main .

#Set the env. variable to use buildkit
#export DOCKER_BUILDKIT=1

RUN --mount=type=cache,target=/go/pkg/mod go mod download

RUN --mount=type=cache,target=/root/.cache/go-build go build -o /app/main/databuff

FROM alpine:latest

ENV ADMROOT=appdata/adm/
ENV ADMFNM=StatDesc.json
ENV DSST=Dsstat.json

ENV DATAROOT=data/datavar/
ENV DATANM=data/datavar

ENV LOGROOT=appdata/log/
ENV LOGSTREAM=LogStream.json


ENV SERVURL=:8080

ENV DTVOLROOT=data/
ENV WCHSZ=10


WORKDIR /app
RUN mkdir appdata
RUN mkdir data
COPY --from=build /app/main/databuff .

EXPOSE 8080


CMD  [ "./databuff" ]


