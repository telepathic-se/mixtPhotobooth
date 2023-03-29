#!/bin/bash

# Create a virtual environment with the prompt 'sd'
python -m venv .venv --prompt sd

# Activate the virtual environment
. .venv/bin/activate

# Upgrade pip to the latest version
pip install -U pip

# Install requirements from requirements.txt
pip install -r requirements.txt

cp mixt.service /etc/systemd/system/mixt.service
systemctl daemon-reload
systemctl enable mixt.service
systemctl start mixt.service

#sudo journalctl -u gradio_image_mixer.service -f

# Change directory to the 'scripts' folder
#cd scripts

# Run the uvicorn server with the Gradio Image Mixer app on host 0.0.0.0 and port 4000
#uvicorn gradio_image_mixer:app --host 0.0.0.0 --port 4000