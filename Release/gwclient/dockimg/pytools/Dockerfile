# syntax=docker/dockerfile:1
FROM python:3.10 AS comp

RUN mkdir build

WORKDIR /build

RUN apt-get -y update && apt-get install -y gfortran 

RUN wget http://www.cmake.org/files/v3.15/cmake-3.15.0.tar.gz && \ 
tar xzf cmake-3.15.0.tar.gz 

WORKDIR /build/cmake-3.15.0
RUN ./bootstrap 

RUN make && make install 


CMD [ "sleep", "infinity" ]
