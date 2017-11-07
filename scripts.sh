
# docker run -p 8888:8888  --name cntk-tutorial-notebooks -it microsoft/cntk:2.0-cpu-python3.5 bash -c "source /cntk/activate-cntk && jupyter-notebook --no-browser --port=8888 --ip=0.0.0.0 --notebook-dir=/cntk/Tutorials --allow-root"

start-notebooks()
{
  if [[ $(docker ps -aqf name=cntk-jupyter-notebooks) =~ ^[:space:]*$ ]]; then
    docker run -p 8888:8888  --name cntk-jupyter-notebooks -v $PWD/notebooks:/root/notebooks -it microsoft/cntk:2.1-cpu-python3.5 bash -c "source /cntk/activate-cntk && jupyter-notebook --no-browser --port=8888 --ip=0.0.0.0 --notebook-dir=/root/notebooks --allow-root"
  else
    docker start -ai cntk-jupyter-notebooks
  fi
}
