import random
import numpy as np
from torch.utils.data import Dataset


class PIPE_Dataset(Dataset):
    def __init__(self, dataset, split='train', location_probability = 0.25):
        self.dataset = dataset[split]
        self.keys = ['source_img', 'target_img', 'Instruction_VLM-LLM', 'Instruction_Class', 'Instruction_Ref_Dataset', 'object_location']
        self.location_probability = location_probability
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        # Select a random instruction
        instructions = [self.dataset['Instruction_VLM-LLM'][idx],self.dataset['Instruction_Class'][idx],self.dataset['Instruction_Ref_Dataset'][idx]]
        instruction = random.choice([instr for instr in instructions if instr])

        # Optionally add location with predefined probability
        if random.random() < self.location_probability: instruction += f" at {self.dataset['object_location'][idx]}"

        # Load images (already loaded in the dataset)
        source_img = self.dataset['source_img'][idx]; target_img = self.dataset['target_img'][idx]
        
        # Convert images to numpy arrays
        source_img = np.array(source_img); target_img = np.array(target_img)
        
        return source_img, target_img, instruction