from transformers import T5EncoderModel
from diffusers import DiffusionPipeline
from diffusers.utils import pt_to_pil
import gc 
import torch


def flush():
    gc.collect()
    torch.cuda.empty_cache()


text_encoder = T5EncoderModel.from_pretrained(
    "DeepFloyd/IF-I-XL-v1.0",
    subfolder="text_encoder", 
    device_map="balanced", 
    load_in_8bit=True, 
    variant="8bit"
)


pipe = DiffusionPipeline.from_pretrained(
    "DeepFloyd/IF-I-XL-v1.0", 
    text_encoder=text_encoder, # pass the previously instantiated 8bit text encoder
    unet=None, 
    device_map="balanced"
)

prompt = "a child was playing with pokemon in heart of downtown toronto"

prompt_embeds, negative_embeds = pipe.encode_prompt(prompt)

del text_encoder
del pipe
flush()

pipe = DiffusionPipeline.from_pretrained(
    "DeepFloyd/IF-I-XL-v1.0", 
    text_encoder=None, 
    variant="fp16", 
    torch_dtype=torch.float16, 
    device_map="balanced"
)

generator = torch.Generator().manual_seed(1)

image = pipe(
    prompt_embeds=prompt_embeds,
    negative_prompt_embeds=negative_embeds, 
    output_type="pt",
    generator=generator,
).images


pil_image = pt_to_pil(image)
pipe.watermarker.apply_watermark(pil_image, pipe.unet.config.sample_size)

# pil_image[0]
del pipe
flush()

########### Super Resolution Pt 1 ###########
pipe = DiffusionPipeline.from_pretrained(
    "DeepFloyd/IF-II-L-v1.0", 
    text_encoder=None, # no use of text encoder => memory savings!
    variant="fp16", 
    torch_dtype=torch.float16, 
    device_map="balanced"
)

image = pipe(
    image=image, 
    prompt_embeds=prompt_embeds, 
    negative_prompt_embeds=negative_embeds, 
    output_type="pt",
    generator=generator,
).images

pil_image = pt_to_pil(image)
pipe.watermarker.apply_watermark(pil_image, pipe.unet.config.sample_size)

del pipe
flush()


########### Super Resolution Pt 2 #############

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-x4-upscaler", 
    torch_dtype=torch.float16, 
    device_map="balanced"
)
pil_image = pipe(prompt, generator=generator, image=image).images
pil_image.save('image.png')

del pipe
flush()
