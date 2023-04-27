

## PhotoBooth
use docker: runpod/stable-diffusion:web-automatic-4.0.0

To Run

```bash
git clone https://github.com/telepathic-se/mixtPhotobooth.git
apt install screen
cd mixtPhotobooth
python -m venv .venv --prompt sd
. .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
cd scripts
screen
uvicorn gradio_image_mixer:app --host 0.0.0.0 --port 4000
<ctrl>a + d
screen -r to reattach

```
