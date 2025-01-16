import os
import torchaudio
from torch.utils.data import Dataset

from datasets import Dataset
import torchaudio.transforms as T
from transformers import Wav2Vec2Processor

import torch

# Ensure that the device is set to CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable CUDA (GPU)
torch.device('cpu')  # Set PyTorch to use the CPU


def load_audio_file(file_path):
    """
    Loads an audio file and returns its waveform.
    """
    waveform, sample_rate = torchaudio.load(file_path)
    return waveform, sample_rate

def create_huggingface_dataset(root_dir):
    """
    Converts an audio dataset (root_dir) into a Hugging Face Dataset.
    """
    # Prepare data (audio files and labels)
    audio_files = []
    labels = []
    class_labels = sorted(os.listdir(root_dir))  # List of subfolders (class names)
    
    for idx, class_label in enumerate(class_labels):
        class_folder = os.path.join(root_dir, class_label)
        for file_name in os.listdir(class_folder):
            if file_name.endswith('.wav') or file_name.endswith('.mp3'):
                file_path = os.path.join(class_folder, file_name)
                audio_files.append(file_path)
                labels.append(idx)  # Assign numeric label based on folder index
    
    # Load the audio data and process it using the processor

    def preprocess_audio_data(file_path):
        """
        Preprocesses an audio file: load, resample, and extract features for Hugging Face model.
        """
        # Load the audio file
        waveform, sample_rate = torchaudio.load(file_path)
        
        # Resample to the required 16kHz if needed
        target_sample_rate = 16000
        if sample_rate != target_sample_rate:
            resampler = T.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
            waveform = resampler(waveform)
            sample_rate = target_sample_rate
        
        # Initialize the processor
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
        
        # Preprocess waveform to the format expected by Hugging Face models
        inputs = processor(waveform.squeeze().numpy(), return_tensors="pt", sampling_rate=sample_rate, device='cpu')
        
        return inputs


    # Prepare data in the Hugging Face format
    data = {"audio": audio_files, "label": labels}
    dataset = Dataset.from_dict(data)
    
    # Add preprocessing step for the audio data
    def preprocess_function(examples):
        audio_paths = examples["audio"]
        audio_inputs = [preprocess_audio_data(path) for path in audio_paths]
        return {"input_values": [item["input_values"] for item in audio_inputs]}

    dataset = dataset.map(preprocess_function, batched=True)
    return dataset

class AudioDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the audio files, organized in subfolders (class labels).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        
        self.audio_files = []
        self.labels = []
        self.class_labels = sorted(os.listdir(root_dir))  # List of subfolders (class names)
        
        for idx, class_label in enumerate(self.class_labels):
            class_folder = os.path.join(root_dir, class_label)
            for file_name in os.listdir(class_folder):
                if file_name.endswith('.wav') or file_name.endswith('.mp3'):
                    file_path = os.path.join(class_folder, file_name)
                    self.audio_files.append(file_path)
                    self.labels.append(idx)  # Assign numeric label based on folder index

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        label = self.labels[idx]
        
        # Load the audio file
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Apply transformation if needed
        if self.transform:
            waveform = self.transform(waveform)
        
        # Return the processed inputs and the label
        return waveform, label
