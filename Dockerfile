FROM rayproject/ray:latest

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


ENV VIRTUAL_ENV=/home/ray/venv
RUN python3.8 -m venv $VIRTUAL_ENV 
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN pip install --upgrade pip
COPY . .
RUN pip3 install --no-cache-dir -r requirements.txt
RUN poetry install
RUN c4v --help

ENTRYPOINT [ "c4v" ]
