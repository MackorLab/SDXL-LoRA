word_0 = "Загрузка библиотеки зависимостей Payton, пожалуйста, подождите!"
print(word_0.upper())


word_1 = "Общее время старта займет приблизительно 5 минут (будем загружать несколько моделей для более правильной интерпретации вашего запроса и повышения качества входного изображения), пожалуйста, ожидайте!"
print(word_1.upper())







art_dia = """


                                     ...
                           ......::::::::::::::::::......
                         ..:::----========--==-===----::::....
                      ...::---==++=-:..       ..::-======--:::...
              ..::. ...::-===+=-..   . ........     ..:-=+==---::..
           .:-=++:...::--===:........:::::::::::.........:=++==--::..
          .-+***-...::--+=:.....:::----=-===--=---::::......=**+==-::..:-:..
        ..-=+**-...:--=+-....::--==+*++*******+++====---:::..:=++==--:.-**+=-.
        .:-=++=:..:-==+-..:::-=+*%%%%%@@@@@%@%%%%%##**++=--:::.-++==--:-*##*+-:.
        .:-=++-..:-==+-.::-=+*%@@@@@@%%#**++**#%%%@@@@@%#*=---::-++===-:*###*+-:..
        .:-===:.::-=+=::-=+*%@@+#@@@*=---::::--=+*##@@%@@@@*+=--:-+++=--=###*+=-:.
         .-==-..:-=++=:-=+#%@@@%@@@@*:.........::-=*@#%@@@@@#+=---++++=--*#**+=-:
          .-=:..:-=+=-:=+#%%#%@@@@@*:..:::--==++-::-%@@@@@@@@#+==-=+++=--+**++=-
              .:---==--=+#%*=--=-:. ...**#%#*%@@@::::=*%%%%%@%++===+*++=-+***+=.
              .:------==*##=:..     ...=#%###%@@#-:::..::-+*%#++++++**+=-+#+-
              .:-----==+*%*:.    .....::-=++****+=--::::::-=#%+*+*+*+**=--
              .:----=+++*%*:.......::::--==+++++++===------=*%**+******+-:
              .:----=+++*#%=--------++====++****++++++++++++##*#*******+-.
               :---=++***#%%*+++++++==+**##%%%%%###*****###%%###*##****=-
               .--==+++**##%@##*******###########%%%%%%%%%@%%########*+=:
                :-=++++**##%%@@%%#######%%%%%%%%%%%%%%%@@@%%%%#%#####*+-
                 :-+***###%%%@@@@@@%%%%%%%%%%%%%%%@@@@@@@%%%%%%%#####+=.
                 .:=+**##%%%%@@@@@@@@@@@%%%%@@@@@@@@@@@@@%%%@%%%%##*=::
                ..:==+**##%%%%%@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%%%%#*=-:..
              ..::-=++*##%%###%%@@@@@@@@@@@@@@@@@@@@@@@@%#%@@@@%%#*+=-:..
             ..::--=+**#%%@#*+*#%%%@@@@@@@@@@@@@@@@@@@%#+*@@@%%%##*++=-::..
            ..::--=++**#%%@@#*++*###%%%%%%@@%@@@@@@@%%###@@@@%%##**++==--:..
           ..::--==+**##%%%@@#*+*****######%%%%%%%%%%#%%@@@@@%%##**+++==--:..
          ...::-==++**##%%@@@%*+++++++++******#######%%@@@@@%%%#***+++===--:..
         ...::--=++***##%%@@@#*++===--======+++++***###%%@@@@%##***+++===---:..
         ..::--==++*###%%%@@%*+=---:-------=====++++***#%@@@@%%##**+++=====-::..
        ...:---=++**##%%%@@%*+=::::::-------========+++*##%@@@%%#**+++====---::.
       ...::--=+++**##%%@@%*+-::::::::----===========+++**#%%#%%#**+++====---::..
       ...::-==++**##%%@@%*=-:::::::--==++*****+++=====++**##*#%##*++=====---:::.
      ...::--==++**#%%%@@#+-::::::--+**######%%%##*+++==+++*##*#%#*++=====-----:..
  """
art = """
───███─███─████─████─███───────────████──███─████─█───█─████─█──█─███─█──█─████─████─████─████
───█────█──█──█─█──█──█────────────█──██──█──█──█─██─██─█──█─██─█──█──█─█──█──█─█──█─█──█─█──█
───███──█──████─████──█─────███────█──██──█──████─█─█─█─█──█─█─██──█──██─────██───██───██───██
─────█──█──█──█─█─█───█────────────█──██──█──█──█─█───█─█──█─█──█──█──█─█───██───██───██───██─
───███──█──█──█─█─█───█────────────████──███─█──█─█───█─████─█──█─███─█──█─██───██───██───██──

"""


from diffusers import DiffusionPipeline
import torch
from compel import Compel, ReturnedEmbeddingsType
from diffusers import DiffusionPipeline, AutoencoderKL
from PIL import Image
from datetime import datetime
from tqdm.auto import tqdm
import os
from random import randint

print(art_dia)
print(art)


vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix",
                                    torch_dtype=torch.float16)
base = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    vae=vae,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
)

base.load_lora_weights("minimaxir/sdxl-wrong-lora")

_ = base.to("cuda")
# base.enable_model_cpu_offload()  # recommended for T4 GPU if enough system RAM

refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=base.text_encoder_2,
    vae=base.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)

_ = refiner.to("cuda")






compel_base = Compel(tokenizer=[base.tokenizer, base.tokenizer_2] , text_encoder=[base.text_encoder, base.text_encoder_2], returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED, requires_pooled=[False, True])
compel_refiner = Compel(tokenizer=refiner.tokenizer_2 , text_encoder=refiner.text_encoder_2, returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED, requires_pooled=True)

high_noise_frac = 0.8



import gradio as gr
import torch
import numpy as np
from PIL import Image

def gen_image(source_prompt, negative_prompt, height, width, cfg=13, seed=-1, webp_output=True):
    if seed < 0:
        seed = np.random.randint(0, 10**8)
        print(f"Seed: {seed}")

    prompt = source_prompt
    negative_prompt = "wrong"

    conditioning, pooled = compel_base(prompt)
    conditioning_neg, pooled_neg = compel_base(negative_prompt) if negative_prompt is not None else (None, None)
    generator = torch.Generator(device="cuda").manual_seed(seed)

    latents = base(prompt_embeds=conditioning,
                   pooled_prompt_embeds=pooled,
                   negative_prompt_embeds=conditioning_neg,
                   negative_pooled_prompt_embeds=pooled_neg,
                   height=height,
                   width=width,
                   guidance_scale=cfg,
                   denoising_end=high_noise_frac,
                   generator=generator,
                   output_type="latent",
                   cross_attention_kwargs={"scale": 1.}
                   ).images

    conditioning, pooled = compel_refiner(prompt)
    conditioning_neg, pooled_neg = compel_refiner(negative_prompt) if negative_prompt is not None else (None, None)
    generator = torch.Generator(device="cuda").manual_seed(seed)

    images = refiner(
        prompt_embeds=conditioning,
        pooled_prompt_embeds=pooled,
        negative_prompt_embeds=conditioning_neg,
        negative_pooled_prompt_embeds=pooled_neg,
        guidance_scale=cfg,
        denoising_start=high_noise_frac,
        image=latents,
        generator=generator,
    ).images

    return images[0]

def convert_seed_to_int(seed):
    if seed == 'random':
        return -1
    return int(seed)

inputs = [
        gr.inputs.Textbox(label='Что вы хотите, чтобы ИИ генерировал?'),
        gr.inputs.Textbox(label='Что вы не хотите, чтобы ИИ генерировал?', default='(deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, (mutated hands and fingers:1.4), disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation, wrinkles, old face'),
        gr.Slider(512, 1024, 768, step=128, label='Высота изображения'),
        gr.Slider(512, 1024, 768, step=128, label='Ширина изображения'),
        gr.Slider(1, maximum=15, value=7, step=0.1, label='Шкала расхождения'),
        gr.Slider(1, maximum=100, value=25, step=1, label='Количество итераций'),
        gr.Slider(label="Точка старта функции", minimum=1, step=1, maximum=9999999999999999, randomize=True)
]

outputs = gr.outputs.Image(type="pil")

def inference(source_prompt, negative_prompt, height, width, cfg, iterations, seed):
    seed = convert_seed_to_int(seed)
    return gen_image(source_prompt, negative_prompt, height, width, cfg, seed)

interface = gr.Interface(
    fn=inference,
    inputs=inputs,
    outputs=outputs,
    title='DIAMONIK7777 - txt2img  - SDXL - LoRA - Refiner - Compel',
    description="<p style='text-align: center'>Будь в курсе обновлений <a href='https://vk.com/public221489796'>ПОДПИСАТЬСЯ</a></p>",
    article="<br><br><p style='text-align: center'>Генерация индивидуальной модели с собственной внешностью <a href='https://vk.com/im?sel=-221489796'>ПОДАТЬ ЗАЯВКУ</a></p><br><br><br><br><br>",
)

interface.launch(debug=True, max_threads=True, share=True, inbrowser=True)


