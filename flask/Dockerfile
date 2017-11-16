
FROM microsoft/cntk:2.1-cpu-python3.5

# cribbed heavily from https://github.com/Azure-App-Service/python/blob/master/3.6.1/Dockerfile

# ssh
ENV SSH_PASSWD "root:Docker!"

SHELL ["/bin/bash", "-c"]

# install nginx / install and configure openssh / clean up
RUN set -ex \
	&& apt-get update \
	&& apt-get install -y -V --no-install-recommends nginx \
	&& rm -r /var/lib/apt/lists/* \
	&& apt-get update \
	&& apt-get install -y --no-install-recommends openssh-server \
	&& echo "$SSH_PASSWD" | chpasswd \
	&& source /cntk/activate-cntk \
	&& pip install flask \
	&& apt-get purge -y -V -o APT::AutoRemove::RecommendsImportant=false --auto-remove $tools $buildTimeDeps \
	&& apt-get autoremove -y

# Add Tini
ENV TINI_VERSION v0.16.1
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /tini
RUN chmod +x /tini

# Configure ports
EXPOSE 2222 80

# Copy files
COPY app/ /root/app
COPY config/nginx-default-site /etc/nginx/sites-available/default
COPY config/sshd_config /etc/ssh/
COPY entrypoint.sh /usr/local/bin

# Adjust permissions
RUN chmod u+x /usr/local/bin/entrypoint.sh

# flask config
ENV LC_ALL "C.UTF-8"
ENV LANG "C.UTF-8"
ENV FLASK_APP "app"

WORKDIR /home

# Set entrypoint and command
ENTRYPOINT ["/tini", "--", "entrypoint.sh"]
CMD ["start-service"]
