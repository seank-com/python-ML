#!/bin/bash -i

mv -v -n -t /home /root/app

if [ "$1" == "start-service" ]; then

  python --version
  pip --version
  echo "INFO: installed packages ..."
  pip freeze

  echo "INFO: environment ..."
  env

  echo "INFO: processes ..."
  ps -ax

  echo "INFO: starting SSH ..."
  service ssh start

  # setup nginx log dir
  # http://nginx.org/en/docs/ngx_core_module.html#error_log
  sed -i "s|access_log /var/log/nginx/access.log;|access_log stdout;|g" /etc/nginx/nginx.conf
  sed -i "s|error_log /var/log/nginx/error.log;|error_log stderr;|g" /etc/nginx/nginx.conf

  echo "INFO: starting nginx ..."
  nginx #-g "daemon off;"

  echo "INFO: starting flask"
  cd /home/app
  flask run
else
  echo "INFO: running alternate command"
  exec "$@"
fi
