# python-ML

Operationalizing Machine Learning from Python in Docker containers

## Links

Based on the [Microsoft CNTK](https://docs.microsoft.com/en-us/cognitive-toolkit/) using info gleaned from the [Cortana Intelligence Gallery](https://gallery.cortanaintelligence.com/)

## CNTK Container
Run the following to import commands

```
$ docker pull microsoft/cntk:2.1-cpu-python3.5
$ source scripts.sh
```

### commands

```start-notebooks``` - Starts or restarts docker container

## Flask container

Run the following to build the containers

```
cd flask
docker build -t imagerec .
```

### Configure Azure

Go to the [Azure Portal](http://portal.azure.com)
- Select Resource groups
  - Click Add
  - Resource group name: imagerec-webapp
  - Resource group location: West US 2
  - Click Create
- Click Go to resource group
  - Click Add
  - enter Webapp
  - Select Web App for Containers
  - Click Create
    - App name: imagerec
    - Resource Group: use existing and select imagerec-webapp
    - Click App Service plan/location
      - Click Create New
        - App Service plan: imagerec-appsvc-plan
        - Location: West US 2
        - Pricing tier: S1 Standard
      - Click OK
    - Click Configure container
      - Click Azure Container Service
        - Registry: SeanK
        - Image: imagerec
        - Tag: latest
      - Click OK
    - Click Create
  - Click Go to resource
- Click Application Settings
  - Always On: On
  - ARR Affinity: On
  - WEBSITES_ENABLE_APP_SERVICE_STORAGE: true
  - Click Save
- Click Docker Container
  - Continuous Deployment: On
  - Click Save

### Deploy to Azure

From the azure cli run the following

```bash
# password can be found under Access keys on the Azure Container Registry
$ docker login seank.azurecr.io -u SeanK -p password
$ docker tag imagerec seank.azurecr.io/imagerec:latest
$ docker push seank.azurecr.io/imagerec:latest
```
