# Azure ML Services Workflow

1. Create an [Azure Machine Learning Workspace](https://docs.microsoft.com/en-us/azure/machine-learning/service/quickstart-get-started)
  _a. If you don't see the **Open Azure Notebooks** button go back to the **Overview** blade and select the **Get Started in Azure Notebooks** button instead._
  _b. You only need to run the first page in the notebook you create in the **Run the notebook** step._
2. [Tutorial (part 1): Train an image classification model with Azure Machine Learning service](https://docs.microsoft.com/en-us/azure/machine-learning/service/tutorial-train-models-with-aml)
  _a. Don't bother cloning the notebook, create a new project in [Azure Notebooks](https://notebooks.azure.com) and just cut and paste the code_
  _b. You will need to download the ```config.json``` file from the previous notebook and then upload it to this project before running the ```ws = Workspace.from_config()``` line
  _c. You can skip the part about creating an AMICompute cluster (sounds expensive) and then re-jigger the **Create a training script** to just save the model you trained locally._
  _d. OK maybe the cluster wasn't expensive afterall since it autoscales to 0 when not in use so we can do that next time_
3. [Deploy models with the Azure Machine Learning service](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-deploy-and-where)
  _a. since the jupyter notebook for the previous step has all the pre-requisites, just keep using it (obviously you will need to change ```model_path``` in the first step)_
  _b. also I got an error about tags being in a dict, so I just removed that parameter_
4. [Tutorial: Deploy Azure Machine Learning as an IoT Edge module (preview)](https://docs.microsoft.com/en-us/azure/iot-edge/tutorial-deploy-machine-learning)
  _a. don't be fooled by the sample IP address in the **Disable process identification** step, you need to use the host IP address. Run ```ifconfig``` and look for the one under ```en01```_
  _b. Don't be tempted to use **Azure Machine Learning Module** when you add the module, you must use **IoT Edge Module**_

