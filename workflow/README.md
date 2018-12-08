# AI-Workflow
Notes and Workflows for Machine Learning

### Build

```bash

$ cd python

# for cpu
$ docker build -f Dockerfile-cpu -t cpu-ai-workflow .

#for gpu
$ docker build -f Dockerfile-gpu -t gpu-ai-workflow .
```

### Run Locally

To Start the container

```bash

# Make sure the external port used is set correctly for the host machine. -p external:container
# for cpu
$ mkdir home
$ docker run -d --rm -v $PWD/home:/home -p 80:80 --name cpu-ai-workflow cpu-ai-workflow

#for gpu
$ nvidia-docker run -d --init --rm -v $PWD/notebooks:/root/notebooks -p 80:80 --name gpu-ai-workflow gpu-ai-workflow
```

On windows the command looks something like this

```cmd
C:\> docker run -d --init --rm -v C:\Users\Redcley\dev\AI-Workflow\python\notebooks:/root/notebooks -p 80:80 --name cpu-ai-workflow cpu-ai-workflow
```


To stop stop the container

```bash

# for cpu
$ docker stop cpu-ai-workflow

# for gpu
$ docker stop cpu-ai-workflow
```

To attach a bash shell to a running container

```bash

# for cpu
$ docker exec -it cpu-ai-workflow bash

# for gpu
$ docker exec -it gpu-ai-workflow bash
```

### Utility Fix-ups

To replace CrLf with Lf before adding to repo

```

perl -i.org -pe 's/\r\n/\n/' python/notebooks/AI-Workflow/DataSets/partitions.csv
```

### Configure Azure

Go to the [Azure Portal](http://portal.azure.com)
- Select Resource groups
  - Click Add
  - Resource group name: ai-workflow-webapp
  - Resource group location: West US 2
  - Click Create
- Click Go to resource group
  - Click Add
  - enter Webapp
  - Select Web App for Containers
  - Click Create
    - App name: ai-workflow
    - Resource Group: use existing and select ai-workflow-webapp
    - Click App Service plan/location
      - Click Create New
        - App Service plan: ai-workflow-appsvc-plan
        - Location: West US 2
        - Pricing tier: S1 Standard
      - Click OK
    - Click Configure container
      - Click Azure Container Service
        - Registry: SeanK
        - Image: cpu-ai-workflow
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
$ docker tag cpu-ai-workflow seank.azurecr.io/cpu-ai-workflow:latest
$ docker push seank.azurecr.io/cpu-ai-workflow:latest
```


### Helpful Reference Material

- The [Pyramid Notebook](https://github.com/websauna/pyramid_notebook) project first hinted that WebSockets would need to be proxied
- [Azure WebApps for Containers](https://docs.microsoft.com/en-us/azure/app-service/containers/quickstart-custom-docker-image) documentation
- [WSGI](https://uwsgi-docs.readthedocs.io/en/latest/WSGIquickstart.html#installing-uwsgi-with-python-support) may or maynot prove to be useful
- [WebSockets in WireShark](https://wiki.wireshark.org/WebSocket) helpful for debugging
- [NGINX as a WebSocket Proxy](https://www.nginx.com/blog/websocket-nginx/)
- [WebScokets proxying on nginx](http://nginx.org/en/docs/http/websocket.html)
- [Running Jupyter notebooks](https://jupyter-notebook.readthedocs.io/en/stable/public_server.html#running-a-public-notebook-server)
- [nginx by Stephen Corona](https://www.safaribooksonline.com/library/view/nginx/9781491924761/ch04.html) demonstrates an alternate nginx config for proxying websocket upgrade requests.
- [comment #17 of this Chromium issue](https://bugs.chromium.org/p/chromium/issues/detail?id=148908#c17) documents a cool hidden feature of chromium to diagnose issues without Wireshark.

After we are done, investigate [AKS](https://docs.microsoft.com/en-us/azure/aks/tutorial-kubernetes-scale)

##################################################

docker run -ti --rm -p 80:80 cpu-ai-workflow bash

pip install flask

export LC_ALL=C.UTF-8
export LANG=C.UTF-8
export FLASK_APP=index.py
python -m flask run -p 8080

cp -r --update --backup -t /home /root/notebooks
find . -name "*~" -delete
