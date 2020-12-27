# The Google App Engine python runtime is Debian Jessie with Python installed
# and various os-level packages to allow installation of popular Python
# libraries. The source is on github at:
# https://github.com/GoogleCloudPlatform/python-docker
FROM gcr.io/google_appengine/python

RUN apt-get update
RUN apt-get -y install binutils libproj-dev gdal-bin libpq-dev python-dev libxml2-dev libxslt1-dev build-essential libssl-dev libffi-dev

RUN virtualenv -p python3 /env
ENV PATH /env/bin:$PATH

RUN /env/bin/pip install --upgrade pip && /env/bin/pip install -r /requirements.txt

RUN python3 setup.py
RUN python3 server.py
