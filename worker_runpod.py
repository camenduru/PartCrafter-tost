import os
import time
import sys
import types
import importlib.util
import zipfile
from typing import Union, Optional

import torch
import torch.nn.functional as F
import numpy as np
import trimesh
from PIL import Image
from torchvision import transforms
from accelerate.utils import set_seed

from src.utils.data_utils import get_colored_mesh_composition
from src.utils.render_utils import (
    render_views_around_mesh, render_normal_views_around_mesh,
    make_grid_for_images_or_videos, export_renderings
)
from src.pipelines.pipeline_partcrafter import PartCrafterPipeline

import json, requests, random, runpod
from urllib.parse import urlsplit

def download_file(url, save_dir, file_name):
    os.makedirs(save_dir, exist_ok=True)
    file_suffix = os.path.splitext(urlsplit(url).path)[1]
    file_name_with_suffix = file_name + file_suffix
    file_path = os.path.join(save_dir, file_name_with_suffix)
    response = requests.get(url)
    response.raise_for_status()
    with open(file_path, 'wb') as file:
        file.write(response.content)
    return file_path

device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = "pretrained_weights/RMBG-2.0"

def tensor2pil(image: torch.Tensor) -> Image.Image:
    return Image.fromarray((image.squeeze().cpu().numpy() * 255).clip(0, 255).astype(np.uint8))

def pil2tensor(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.asarray(image).astype(np.float32) / 255.0).unsqueeze(0)

def load_rmbg_model():
    import safetensors.torch

    def dynamic_import(module_name: str, file_path: str):
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)
        return mod

    config_module = dynamic_import("BiRefNetConfig", os.path.join(MODEL_DIR, "BiRefNet_config.py"))
    birefnet_code = open(os.path.join(MODEL_DIR, "birefnet.py")).read().replace(
        "from .BiRefNet_config import BiRefNetConfig", "from BiRefNetConfig import BiRefNetConfig"
    )

    birefnet_module = types.ModuleType("birefnet_model")
    exec(birefnet_code, birefnet_module.__dict__)

    for attr in dir(birefnet_module):
        cls = getattr(birefnet_module, attr)
        if isinstance(cls, type):
            try:
                model = cls(config_module.BiRefNetConfig)
                break
            except Exception:
                continue

    model.load_state_dict(safetensors.torch.load_file(os.path.join(MODEL_DIR, "model.safetensors")))
    model.eval().to(device)
    return model

def remove_background_rmbg(image: Image.Image, model, resolution: int = 1024, sensitivity: float = 1.0):
    if model is None:
        raise ValueError("RMBG model must be loaded")

    original_size = image.size
    transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        if isinstance(output, dict) and 'logits' in output:
            mask = output['logits']
        elif isinstance(output, list):
            mask = output[-1]
        else:
            mask = output
        mask = torch.clamp(mask.sigmoid().squeeze() * (1 + (1 - sensitivity)), 0, 1)

    resized_mask = F.interpolate(mask[None, None], size=original_size[::-1], mode='bilinear').squeeze()
    alpha = (resized_mask.cpu().numpy() * 255).astype(np.uint8)

    rgba = np.dstack((np.array(image.convert("RGB")), alpha))
    return Image.fromarray(rgba), Image.fromarray(alpha)

@torch.no_grad()
def generate_parts_from_image(
    image_path: Union[str, Image.Image],
    num_parts: int = 4,
    seed: int = 0,
    num_tokens: int = 1024,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.0,
    max_num_expanded_coords: int = int(1e9),
    use_flash_decoder: bool = False,
    remove_bg: bool = False,
    render: bool = False,
    dtype: torch.dtype = torch.float16,
) -> str:
    assert 1 <= num_parts <= 16, "num_parts must be between 1 and 16"

    set_seed(seed)

    # Create temp folder inside PartCrafter
    temp_dir = os.path.join(os.getcwd(), "temp_output")
    os.makedirs(temp_dir, exist_ok=True)

    # Load models
    pipe = PartCrafterPipeline.from_pretrained("pretrained_weights/PartCrafter").to(device, dtype)
    rmbg_model = load_rmbg_model() if remove_bg else None

    # Load and optionally process image
    input_image = Image.open(image_path).convert("RGB") if isinstance(image_path, str) else image_path
    img_pil, _ = remove_background_rmbg(input_image, rmbg_model) if remove_bg else (input_image, None)

    # Run pipeline
    outputs = pipe(
        image=[img_pil] * num_parts,
        attention_kwargs={"num_parts": num_parts},
        num_tokens=num_tokens,
        generator=torch.Generator(device=device).manual_seed(seed),
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        max_num_expanded_coords=max_num_expanded_coords,
        use_flash_decoder=use_flash_decoder,
    ).meshes

    outputs = [m or trimesh.Trimesh(vertices=[[0, 0, 0]], faces=[[0, 0, 0]]) for m in outputs]
    for i, mesh in enumerate(outputs):
        mesh.export(os.path.join(temp_dir, f"part_{i:02}.glb"))

    merged_mesh = get_colored_mesh_composition(outputs)
    merged_mesh.export(os.path.join(temp_dir, "object.glb"))

    # Optional rendering
    if render:
        num_views, radius, fps = 36, 4, 18
        imgs = render_views_around_mesh(merged_mesh, num_views=num_views, radius=radius)
        normals = render_normal_views_around_mesh(merged_mesh, num_views=num_views, radius=radius)
        grid = make_grid_for_images_or_videos([[img_pil] * num_views, imgs, normals], nrow=3)

        export_renderings(imgs, os.path.join(temp_dir, "rendering.gif"), fps=fps)
        export_renderings(normals, os.path.join(temp_dir, "rendering_normal.gif"), fps=fps)
        export_renderings(grid, os.path.join(temp_dir, "rendering_grid.gif"), fps=fps)

        imgs[0].save(os.path.join(temp_dir, "rendering.png"))
        normals[0].save(os.path.join(temp_dir, "rendering_normal.png"))
        grid[0].save(os.path.join(temp_dir, "rendering_grid.png"))

    # Create ZIP file in PartCrafter directory
    zip_path = os.path.join(os.getcwd(), "PartCrafter.zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(temp_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, start=temp_dir)
                zipf.write(file_path, arcname)

    import shutil
    shutil.rmtree(temp_dir)
    
    return zip_path

@torch.inference_mode()
def generate(input):
    try:
        values = input["input"]

        input_image = values['input_image'] 
        input_image = download_file(url=input_image, save_dir='/content', file_name='input_image')
        num_parts = values['num_parts']
        remove_bg = values['remove_bg']
        render = values['render']
        seed = values['seed']
        if seed == 0:
            random.seed(int(time.time()))
            seed = random.randint(0, 4294967295)

        result = generate_parts_from_image(image_path=input_image, num_parts=num_parts, remove_bg=remove_bg, render=render, seed=seed)
        
        notify_uri = values['notify_uri']
        del values['notify_uri']
        notify_token = values['notify_token']
        del values['notify_token']
        discord_id = values['discord_id']
        del values['discord_id']
        if(discord_id == "discord_id"):
            discord_id = os.getenv('com_camenduru_discord_id')
        discord_channel = values['discord_channel']
        del values['discord_channel']
        if(discord_channel == "discord_channel"):
            discord_channel = os.getenv('com_camenduru_discord_channel')
        discord_token = values['discord_token']
        del values['discord_token']
        if(discord_token == "discord_token"):
            discord_token = os.getenv('com_camenduru_discord_token')
        job_id = values['job_id']
        del values['job_id']
        with open(result, 'rb') as file:
            response = requests.post("https://upload.tost.ai/api/v1", files={'file': file})
        response.raise_for_status()
        result_url = response.text
        notify_payload = {"jobId": job_id, "result": result_url, "status": "DONE"}
        web_notify_uri = os.getenv('com_camenduru_web_notify_uri')
        web_notify_token = os.getenv('com_camenduru_web_notify_token')
        if(notify_uri == "notify_uri"):
            requests.post(web_notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
        else:
            requests.post(web_notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
            requests.post(notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": notify_token})
        return {"jobId": job_id, "result": result_url, "status": "DONE"}
    except Exception as e:
        error_payload = {"jobId": job_id, "status": "FAILED"}
        try:
            if(notify_uri == "notify_uri"):
                requests.post(web_notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
            else:
                requests.post(web_notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
                requests.post(notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": notify_token})
        except:
            pass
        return {"jobId": job_id, "result": f"FAILED: {str(e)}", "status": "FAILED"}
    finally:
        if os.path.exists(result):
            os.remove(result)

runpod.serverless.start({"handler": generate})