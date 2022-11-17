ARG PYTHON="3.9"
ARG PYTORCH="1.12"
ARG CUDA="11.6.1"

FROM pytorchlightning/pytorch_lightning:base-cuda-py${PYTHON}-torch${PYTORCH}-cuda${CUDA}

RUN apt update && apt install ssh -y
RUN apt-get update && apt-get install libgl1

# authorize SSH connection with root account
RUN sed -i 's/#\s*PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# change root password to 'mypassword' <-- PICK SOME OTHER PASSWORD HERE
RUN echo "root:password" | chpasswd

# Necessary configuration for sshd
RUN mkdir /var/run/sshd

# Install shallowmind
RUN git clone https://github.com/CharonWangg/shallowmind.git /shallowmind
WORKDIR /shallowmind
RUN pip install --no-cache-dir -e .

# Start sshd at container spin-up time
CMD service ssh start && bash