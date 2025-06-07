# Paint by Inpaint: Learning to Add Image Objects by <br> Removing Them First
Welcome to the official repository for our paper!

We are happy to announce that our paper has been **accepted to CVPR 2025**! 🎉  

<img src=figures/teaser.png width="80%" alt="Teaset Figure">

## Resources

- 💻 **Project Page**: For more details, visit the official [project page](https://rotsteinnoam.github.io/Paint-by-Inpaint/).

- 📝 **Read the Paper**: You can find the paper [here](https://arxiv.org/abs/2404.18212).

- 🎥 **Watch the Video**: Check out the project video on [YouTube](https://www.youtube.com/embed/Zhj1zkrYrcY).

- 🚀 **Try Our Demo**: Explore our models trained with the PIPE dataset, now available on [Huggingface Spaces](https://huggingface.co/spaces/paint-by-inpaint/demo).

- 🏋️‍♂️ **Use Trained Models**:  See how to use our trained model weights in the [Trained Models](#trained-models) section.

- 🗂️ **Use the PIPE Dataset**: For more details, see the [PIPE Dataset](#pipe-dataset) section.


## Trained Models

We release our trained models:
- [addition-base-model](https://huggingface.co/paint-by-inpaint/add-base): Trained on the PIPE dataset, specifically designed for object addition.
- [addition-finetuned-model](https://huggingface.co/paint-by-inpaint/add-finetuned-mb): The addition-base-model fine-tuned on a MagicBrush addition subset.
- [general-base-model](https://huggingface.co/paint-by-inpaint/general-base): Trained on the combined PIPE and InstructPix2Pix datasets, intended for general editing.
- [general-finetuned-model](https://huggingface.co/paint-by-inpaint/general-finetuned-mb): The general-base-model fine-tuned on the full MagicBrush dataset.

Our models are simple to run using the InstructPix2Pix pipeline:
```python
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
import torch
import requests
from io import BytesIO
from PIL import Image

model_name = "paint-by-inpaint/add-base"  # addition-base-model
# model_name = "paint-by-inpaint/add-finetuned-mb"  # addition-finetuned-model
# model_name = "paint-by-inpaint/general-base"  # general-base-model
# model_name = "paint-by-inpaint/general-finetuned-mb"  # general-finetuned-model

diffusion_steps = 50
device = "cuda"
image_url = "https://paint-by-inpaint-demo.hf.space/file=/tmp/gradio/99cd3a15aa9bdd3220b4063ebc3ac05e07a611b8/messi.jpeg"
image = Image.open(BytesIO(requests.get(image_url).content)).resize((512, 512))

pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_name, torch_dtype=torch.float16, safety_checker=None).to(device)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

# Generate the modified image
out_images = pipe(
    "Add a royal silver crown", 
    image=image, 
    guidance_scale=7, 
    image_guidance_scale=1.5, 
    num_inference_steps=diffusion_steps,
    num_images_per_prompt=1
).images

```

## PIPE Dataset
The PIPE (Paint by Inpaint Editing) dataset is a novel resource for image editing, specifically designed to enhance the process of seamlessly adding objects to images by following instructions. This extensive dataset, comprising approximately 1 million image pairs, includes both object-present and object-removed versions of images with over 1400 distinct object classes along with object addition language instructions with countless different object attributes. Unlike other editing datasets, PIPE consists of natural target images and maintains consistency between source and target images by construction.

###   Download
The training and testing dataset can be found on Hugging Face [here](https://huggingface.co/datasets/paint-by-inpaint/PIPE), along with a full explanation about the dataset and its use. It is important to note the instruction loading—the use of different types of instructions (class-based, VLM-LLM-based, and reference-based), and the use of the object location.

####   Load Example
```python
from datasets import load_dataset
from torch.utils.data import DataLoader
from dataset.dataset import PIPE_Dataset

data_files = {"train": "data/train-*", "test": "data/test-*"}
pipe_dataset  = load_dataset('paint-by-inpaint/PIPE',data_files=data_files)

train_dataset = PIPE_Dataset(pipe_dataset, split='train')
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = PIPE_Dataset(pipe_dataset, split='test')
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
```

### Test Set
The PIPE test set can be downloaded from [google drive](https://drive.google.com/file/d/1lM_K4DFmply7FctneEgly8wbwq3KBdle/view).
It contains 752 image pairs from the COCO dataset—without and with objects—along with object addition instructions, and includes a README file that explains the procedure for loading it.

### Object Masks
The masks used to generate the PIPE train and test dataset are available on [Hugging Face](https://huggingface.co/datasets/paint-by-inpaint/PIPE_Masks).

##  BibTeX
```
@article{wasserman2024paint,
  title={Paint by Inpaint: Learning to Add Image Objects by Removing Them First},
  author={Wasserman, Navve and Rotstein, Noam and Ganz, Roy and Kimmel, Ron},
  journal={arXiv preprint arXiv:2404.18212},
  year={2024}
}
```
