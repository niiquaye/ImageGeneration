FROM ubuntu:20.04

WORKDIR /app
COPY Imagen/* /app/

RUN apt-get update && apt-get install python3-pip -y
RUN pip install huggingface_hub --upgrade

RUN python3 -m pip install cache purge
RUN python3 -m pip install --no-cache-dir -r /app/requirements.txt
# RUN python3 -m pip install --no-cache-dir \
# diffusers~=0.16 \
# transformers~=4.28 \
# safetensors~=0.3 \
# sentencepiece~=0.1 \
# accelerate~=0.18 \
# bitsandbytes~=0.38 \
# torch~=2.0 -q



CMD ["python3", "/app/login.py"]
CMD ["python3", '/app/main.py']
