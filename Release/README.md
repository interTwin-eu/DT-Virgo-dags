
# DT Datastore Services

This repository contains the source code of the services responsible for storing gravitational waves data(gw) .
Additional software tools for infrastructure testing are included. The services have been designed to be executed inside a cluster running Kubernetes. Their Docker images can be found here:

https://hub.docker.com/repositories/romanoa77

## Data Buffer

The Data Buffer is a storage service written in Go. It stores gw data as JSON files and collects metadata about the infrastructure.

### 1. Requirements
Go 1.23.3 or more recent version.
### 2. Configuration
Environment variables are defined inside the Dockerfile. For a local installation the envsetup.sh file can be used.

### 3. Container Setup
Environment variables description. <br><br>


> ADMROOT

Directory containing metadata.The path is expected to be  mountpoint/directory,  mountpoint is where the kubernetes volume is mounted. 
 
>ADMFNM

 Name of the metadata file relating to the gw stored data.

>DSST

Name of the metadata file relating to the datastore state.

>DATAROOT

Directory containing gw data. The path is expected to be  mountpoint/directory.

>DATANM

Directory containing gw data(duplicate).

>LOGROOT

Log directory location.

>LOGSTREAM

Log file name.

>SERVURL

Port where the service listens for connections.

>DTVOLROOT

Directory containing gw data inside the volume.

>WCHSZ

Requests buffer size.

### 4. Kubernetes Setup

Several object configuration files are used: <br><br>

-state.yaml

Defines a statefulset primitive for the service.

-adml.yaml

Defines a volume storing application metadata and the log file.

-gwdataln.yaml

Defines the volume storing gw data.

### 5. Commands
To compile the source first install the required  go libraries, from the main directory

> go mod download

then give the build command

> go build -o databuff

### 6. Directories description

Go is similar to the C programming language so a main function is required. The main directory contains the main function definition with the addition of copies of the application data. The class directory contains the classes definitions. Go does not provide a class data type like in C++. A program can follow the OOP paradigm using
some language features like the struct data type and modules.

### 7. Application data

Inside a container the application will have the following directory tree:

    .
    ├── application/
    │   └── databuff
    ├── datavar/
    │   └── data/
    └── appdata/
        ├── adm/
        │   ├── StatDesc.json
        │   └── Dsstat.json
        └── log/
             └── LogStream.json

- databuff is the application executable
- The datavar and appdata directories are mountpoints for their respective kubernetes volume
- The data directory contains stored gw data
- The adm directory contains metadata relating to the system
- The log directory contains the log file

Regarding the metadata:

-StatDesc.json contains the following fields:

  - n_itm: the number of files stored
  - buff_size: total number of bytes written

-Dsstat.json contains the following fields:

  - dstatus: datastore state. It can assume the OPERATIONAL/FROZEN values
  - user: client identification string
  - token: identity token

The last fields are related to a rudimentary client identification mechanism. 

For a local installation the directory tree is the same as the one inside the Databuffcomp directory.
    
### 8. Endpoints Description

GET methods

>  /stat

The service will send the content of the StatDesc.json file in  the JSON format.

>  /dumpLogF


The service will send the content of the log file in the JSON format.

>  /dstat

The service will send the content of the Dsstat.json file.

>  /dumpF

The service will send a list of the stored files using the JSON format.



POST methods

>  /sendF

This method accepts a JSON file containing gw data. The received data will be stored on disk.

>  /upddsc

The service will update the Dsstat.json file with the content of the received JSON file. This is the the method used by the Datastore logic to freeze the Datastore.

>  /cleanall

This method will unfreeze the datastore. All written data will be stored inside a  directory named as "DAY HOUR MIN", the date refers to when the request has been accepted. 

## Datastore Logic

The Datastore Logic  is a web application written with  Flask. It performs the datastore operations using the functionalities of the Data Buffer service. The service runs inside a gunicorn server.

### 1. Requirements

The following python packages are required: <br> <br>

    blinker==1.7.0
    certifi==2024.2.2
    charset-normalizer==3.3.2
    click==8.1.7
    flask==3.0.3
    idna==3.7
    importlib-metadata==7.1.0
    itsdangerous==2.1.2
    Jinja2==3.1.3
    MarkupSafe==2.1.5
    PyYAML==6.0.1
    requests==2.31.0
    urllib3==2.2.1
    werkzeug==3.0.2
    zipp==3.18.1
    gunicorn==22.0.0

### 2. Configuration

Environment variables are defined inside the Dockerfile. For a local installation the envsetup.sh file can be used. In addition the gunicorn server can be configured
using the gunicorn.conf.py file. See gunicorn documentation https://docs.gunicorn.org/en/22.0.0/settings.html .
### 3. Container Setup
Environment variables description <br><br>

> ENDP_STAT

Name of the Data Buffer endpoint for getting the content of the StatDesc.json file.

> ENDP_DUMP

Name of the Data Buffer endpoint for getting the content of the log file.

> ENDP_DUMPF

Name of the Data Buffer endpoint for getting the list of the stored  gw data files.

> ENDP_SEND

Name of the Data Buffer endpoint used for sending gw data.

> ENDP_DESC

Name of the Data Buffer endpoint for getting the content of the Dsstat.json file.

> ENDP_STAT

Name of the Data Buffer endpoint for updating the content of the StatDesc.json file.

> ENDP_CLEAN

Name of the Data Buffer endpoint for unfreezing the datastore.

> DB_BURL

Data Buffer url.

> MAX_SIZE

Size of the stored gw data required for freezing the datastore.
### 4. Kubernetes Setup

-depl.yaml

Defines a deployment for the service.
### 5. Commands

If you want to launch the service from a local installation use the command

> gunicorn -c /path to gunicorn.conf.py app:create_app('default')

for example:

> gunicorn -c /etc/gunicorn.conf.py app:create_app('default')

### 6. Directory Description

    .
    ├── app/
        ├── main/
        └── static/
        └── templates/

- main contains methods definition for serving requests
- static contains resources used in html pages
- templates contains html pages
     
### 7. Endpoint Description

POST methods

> /flushbuf

Method for cleaning datastore

> /freezeds

Method for freezing datastore

> /stream

Method for sending gw data


GET methods

> /dsdesc

Method for getting the content of the Dsstat.json

> /dumpflog


Method for getting the content of the log file

> /dumpf

Method for getting the list of the stored gw data files

> /

Datastore status page

> /bufstat

Method for getting the content of the StatDesc.json

## GlitchflowAPI
API layer of the infrastructure. Developed using the FastAPI Python framework.

### 1. Requirements
The following python packages are required:

    fastapi==0.111.0
    httpx==0.27.0
    pydantic==2.7.4

### 2. Configuration
Environment variables are defined inside the Dockerfile. For a local installation the envsetup.sh file can be used.
FastAPI runs upon an Uvicorn webserver, see the official documentation for more details https://fastapi.tiangolo.com/deployment/server-workers/#multiple-workers

### 3. Container Setup

> DS_STAT

Name of the Datastore Logic endpoint for getting the content of the StatDesc.json file.


> DS_DUMPF

Name of the Datastore Logic for getting the list of the stored  gw data files.

> DS_SEND

Name of the Datastore Logic endpoint used for sending gw data.

> DS_DESC

Name of the Datastore Logic endpoint for getting the content of the Dsstat.json file.

> DS_FREEZE

Name of the Datastore Logic endpoint for freezzing the datastore.

> DS_FLUSH

Name of the Data Buffer endpoint for unfreezing the datastore.

> DS_BURL

Datastore Logic  url.

> MAX_SIZE

Size of the stored gw data required for freezing the datastore.
### 4. Kubernetes Setup
-deplapi.yaml

Defines a deployment for the service.

### 5. Commands
To launch the service on a local installation use the command

> fastapi run app/main.py --port 8000

### 6. API Description

GET methods

>/stats

Retrieve the content of the StatDesc.json file.

    {"n_itm": NUMBER OF FILES STORED, "buff_size": TOTAL SIZE OF STORED DATA}

>/dumpbuf

Retrieve a list of the stored gwdata files.

    {"nmlist":[ARRAY OF STRINGS]}

>/desc

Datastore status.

    {"resp": OPERATIONAL/FROZEN}

>/dspage

Datastore status page.

POST methods

> /train

Datastore freezing signal. The following JSON file is expected to be sent with the request

    {"user": CLIENT ID,"token": IDENTITY TOKEN}

in case the required batch of data was collected the response will be:

    {"resp":'FROZEN',"n_f": NUMBER OF FILES STORED,"bt_wrt": TOTAL BYTE WRITTEN}

instead if the datastore is still waiting for data

    {"resp":'WAITING'}

> /streamdata

API for sending data. The expected timeseries is:

    {"h":STRAIN ARRAY,"t":[t0,dt]}

for a stored file the response is:

    {"resp":'CREATED',"body": ADDITIONAL INFORMATION}

if the datastore is frozen the response will be:

    {"resp":'FROZEN',"body": ADDITIONAL INFORMATION}

> /cleanbuf    

API for unfreezing the datastore. The following JSON file is expected to be sent with the request   

     {"msg": THE CONTENT IS NOT RELEVANT AT THE MOMENT}

## GWclient

This is a simple python script for sending data from gw public catalogues to the datastore. During platform tests it has been executed from a shell interacting with a pod. 
It relies on the gwpy framework.

### Requirements

    aiohttp==3.10.3
    gwosc==0.7.1
    gwpy==3.0.8
    numpy==1.24.4

### Container setup 

For tests the image of the pod has been my custom docker image gwpyimg:0.1.rc .

### Usage 

Copy the content of the Gw client directory inside a pod. Configure the script using the envsetup.sh file then execute the python script.

### Configuration

> GWID

Gravitational waves identifier.

> DTCID

Interferometer identifier (see gwpy documentation).

> INF
> SUP

Data will be sent as an interval (INF,SUP) around the gps of the gravitational waves. 

> SRV

Url of the API transmitting data.

## Client

A collection of shell scripts for sending requests to the datastore. They can be used like the gwclient script. 

## Airflow DAG

The following API are used:

>train

>stats

A docker image containing the preprocessing software is required.






