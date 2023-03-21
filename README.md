

## PhotoBooth

To Run

```bash
git clone https://github.com/telepathic-se/mixtPhotobooth/stable-diffusion.git
cd stable-diffusion
git checkout 1c8a598f312e54f614d1b9675db0e66382f7e23c
python -m venv .venv --prompt sd
. .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
cd scripts
python gradio_image_mixer.py
```
