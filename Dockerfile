FROM rayproject/ray:3d764d-py38

# Poetry version
ENV POETRY_VERSION='1.1.6'

# Build profile, one of:
#   * production (for production builds)
#   * development (for development builds)
ENV PROFILE = "development" 

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.8 \
    python3-pip \
    python3-venv \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


# Create a new user 
RUN useradd -ms /bin/bash c4v
USER ray
WORKDIR /home/ray
RUN mkdir c4v-py

# Virtualenv to use
# ENV VIRTUAL_ENV=/home/ray/venv
# RUN python3.8 -m venv $VIRTUAL_ENV
# ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN pip install --upgrade pip
COPY . c4v-py/
WORKDIR /home/ray/c4v-py/

RUN ls -la 

# Set up poetry
RUN pip3 install poetry==$POETRY_VERSION
#   No need of virtualenv as docker is already isolated
RUN poetry config virtualenvs.create false              

# Install project
RUN pip3 install --no-cache-dir -r requirements.txt 
RUN poetry install $(test $PROFILE=='production' && echo "--no-dev") --no-interaction --no-ansi
RUN c4v --help

ENTRYPOINT [ "c4v" ]
