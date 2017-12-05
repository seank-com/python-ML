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

### Deploy to Azure

If you haven't created an Azure Container Registry yet

Go to the [Azure Portal](http://portal.azure.com)
- Click "Create a resource"
- Search for "Azure Container Registry"
- Click Azure Container Registry
- Click Create
  - Fill in the entries (most entries are your choice, except the following)
    - Location: should be the same as the webservice below
    - Admin user: Enable
    - SKU: basic is sufficient
    - Check "pin to dashboard"
    - Click Create

From a command prompt run the following

_**Note:** registry, user and password can be found under Access keys on the Azure Container Registry_

```bash
$ docker login registry.azurecr.io -u user -p password
$ docker tag imagerec registry.azurecr.io/imagerec:latest
$ docker push registry.azurecr.io/imagerec:latest
```

### Configure Azure

_**Note:** registry, user and password can be found under Access keys on the Azure Container Registry_

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
        - Registry: registry
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

### Test

```bash
$ curl curl --header "Content-Type:application/octet-stream"  --data-binary @Mike.jpg http://imagerec.azurewebsites.net
```
