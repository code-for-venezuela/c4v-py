FROM rayproject/ray:3d764d-py38

# Poetry version
ENV POETRY_VERSION='1.1.6'

# Build profile, one of:
#   * production (for production builds)
#   * development (for development builds)
# (not in use right now)
ENV PROFILE = "development" 

USER root

# Set up python 3.8
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.8 \
    python3-pip \
    python3-venv \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


USER ray
WORKDIR /home/ray
RUN mkdir c4v-py

# Set up pip & copy content 
RUN pip install --upgrade pip --no-cache-dir
COPY . c4v-py/
WORKDIR /home/ray/c4v-py/

# Set up poetry
RUN pip3 install poetry==$POETRY_VERSION
#   No need of virtualenv as docker is already isolated
RUN poetry config virtualenvs.create false              

# Install project
RUN pip3 install --no-cache-dir -r requirements.txt 
RUN poetry install --no-dev --no-interaction --no-ansi
RUN c4v --help

# remove unnecessary cache
RUN rm -rf ~/.cache

ENTRYPOINT [ "c4v" ]
