FROM tensorflow/tensorflow:latest-gpu-py3

RUN apt update && apt install -y software-properties-common apt-transport-https wget
RUN wget -q https://packages.microsoft.com/keys/microsoft.asc -O- | apt-key add -
RUN add-apt-repository "deb [arch=amd64] https://packages.microsoft.com/repos/vscode stable main"
RUN apt update && apt install libx11-xcb-dev libasound2 code git gitg fish -y
RUN useradd -m tf
RUN echo "tf ALL=NOPASSWD: ALL" >> /etc/sudoers
RUN pip install pylint pytest autopep8 rope
RUN mkdir -p /home/tf/.config/fish && chown tf /home/tf/.config/fish
RUN chsh -s /usr/bin/fish tf
