# Paint by Inpaint: Learning to Add Image Objects by <br> Removing Them First
Welcome to the official repository for our paper!


<img src=figures/teaser.png width="80%" alt="Teaset Figure">

## Resources

- 💻 **Project Page**: For more details, visit the official [project page](https://rotsteinnoam.github.io/Paint-by-Inpaint/).

- 📝 **Read the Paper**: You can find the paper [here](https://arxiv.org/abs/2404.18212).

- 🚀 **Try Our Demo**: Experience our trained models with the PIPE dataset, available on Huggingface Spaces.

- 🗂️ **Use the PIPE Dataset**: For more details, see the [PIPE Dataset](#pipe-dataset) section below.


## Coming Soon
The code and demo are currently being prepared and will be available soon.

Please check back for updates.


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

##  BibTeX
```
@article{wasserman2024paint,
  title={Paint by Inpaint: Learning to Add Image Objects by Removing Them First},
  author={Wasserman, Navve and Rotstein, Noam and Ganz, Roy and Kimmel, Ron},
  journal={arXiv preprint arXiv:2404.18212},
  year={2024}
}
```
