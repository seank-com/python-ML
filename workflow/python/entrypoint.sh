#!/bin/bash

# stage notebooks into /home/notebooks if they are not already there.

mv -v -n -t /home /root/notebooks

if [ "$1" == "jupyter-notebook" ]; then
  source /cntk/activate-cntk

  touch /root/anaconda3/envs/cntk-py35/lib/python3.5/site-packages/easy-install.pth
  pip install --upgrade pip
  pip install --upgrade notebook==5.2.1

  echo "INFO: starting SSH ..."
  service ssh start

  # setup nginx log dir
  # http://nginx.org/en/docs/ngx_core_module.html#error_log
  sed -i "s|access_log /var/log/nginx/access.log;|access_log stdout;|g" /etc/nginx/nginx.conf
  sed -i "s|error_log /var/log/nginx/error.log;|error_log stderr;|g" /etc/nginx/nginx.conf

  echo "INFO: starting nginx ..."
  nginx

  exec "$@"
fi

exec "$@"
