import os
import pandas as pd
import torchaudio
import torch
from torch.utils.data import Dataset
from transformers import WhisperProcessor

def create_file_mapping(transcript_dir, audio_dir):
    """
    Creates a DataFrame mapping transcript files to their corresponding audio files based on filenames.
    
    Args:
        transcript_dir (str): Directory containing transcript text files.
        audio_dir (str): Directory containing audio files.
    
    Returns:
        pd.DataFrame: DataFrame with columns 'Interview', 'Transcript Path', and 'Audio Path'.
    """
    # Get lists of file paths
    transcript_files = [f for f in os.listdir(transcript_dir) if f.endswith('.txt')]
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
    
    # Extract file identifiers from filenames
    transcript_ids = {os.path.splitext(f)[0] for f in transcript_files}
    audio_ids = {os.path.splitext(f)[0] for f in audio_files}
    
    # Determine the intersection of transcript and audio IDs
    common_ids = transcript_ids.intersection(audio_ids)
    
    # Create the mapping dictionary
    file_mapping = {
        'Interview': [],
        'Transcript Path': [],
        'Audio Path': []
    }
    
    for file_id in common_ids:
        transcript_path = os.path.join(transcript_dir, file_id + '.txt')
        audio_path = os.path.join(audio_dir, file_id + '.wav')
        file_mapping['Interview'].append(file_id)
        file_mapping['Transcript Path'].append(transcript_path)
        file_mapping['Audio Path'].append(audio_path)
    
    # Convert the dictionary to a DataFrame
    return pd.DataFrame(file_mapping)


def process_transcripts(paths_df):
    """
    Processes transcript files to extract and format relevant data, and combines them into a single DataFrame.
    
    Args:
        paths_df (pd.DataFrame): DataFrame with columns 'Interview', 'Transcript Path', and 'Audio Path'.
    
    Returns:
        pd.DataFrame: DataFrame containing concatenated and processed transcript data.
    """
    all_transcript_data = []
    
    for _, row in paths_df.iterrows():
        transcript_path = row['Transcript Path']
        
        # Load transcript data with appropriate delimiter and columns
        transcript_df = pd.read_csv(transcript_path, delimiter="\t", usecols=['StTime', 'EnTime', 'Content'])
        
        # Convert times from minutes to milliseconds
        transcript_df['StTime'] *= 1000
        transcript_df['EnTime'] *= 1000
        
        # Standardize content to uppercase
        transcript_df['Content'] = transcript_df['Content'].str.upper()
        
        # Add the identifier column
        transcript_df['Interview'] = row['Interview']
        
        # Append to the list
        all_transcript_data.append(transcript_df)
    
    # Combine all transcript data into a single DataFrame
    return pd.concat(all_transcript_data, ignore_index=True)


class AudioDataset(Dataset):
    def __init__(self, df, transcript_dir, audio_dir, processor, target_sample_rate=16000, target_length_ms=15000, padding_value=0):
        """
        Audio Dataset with Padding for ASR tasks.

        Parameters:
        - df (pd.DataFrame): DataFrame containing the dataset information.
        - transcript_dir (str): Directory containing transcript files.
        - audio_dir (str): Directory containing audio files.
        - processor: Processor for audio and text processing.
        - target_sample_rate (int): Target sample rate for audio segments.
        - target_length_ms (int): Target length for audio segments in milliseconds.
        - padding_value (float): Value to use for padding audio segments.
        """
        self.df = df
        self.transcript_dir = transcript_dir
        self.audio_dir = audio_dir
        self.processor = processor
        self.target_sample_rate = target_sample_rate
        self.target_length_samples = int(target_length_ms * target_sample_rate / 1000)
        self.padding_value = padding_value

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        start_tm = row['StTime']
        end_tm = row['EnTime']
        content = row['Content']
        interview = row['Interview']
        audio_path = os.path.join(self.audio_dir, interview + '.wav')
        
        # Extract and process audio segment
        audio_segment = self.extract_audio_segment(audio_path, start_tm, end_tm)
        audio_tensor = self.processor(audio_segment.squeeze().numpy(), sampling_rate=self.target_sample_rate, return_tensors="pt").input_features
        
        # Tokenize and pad transcription
        encodings = self.processor.tokenizer(
            content,
            return_tensors="pt",
            padding='max_length',
            truncation=True,
            max_length=448
        )
        
        labels = encodings.input_ids.squeeze()
        attention_mask = encodings.attention_mask.squeeze()
        
        return {
            'input_features': audio_tensor.squeeze(),
            'labels': labels,
            'attention_mask': attention_mask
        }
    
    def extract_audio_segment(self, audio_file_path, start_time_ms, end_time_ms):
        """
        Extracts and pads a segment from an audio file based on start and end times.
        
        Parameters:
        - audio_file_path (str): Path to the audio file.
        - start_time_ms (int): Start time in milliseconds.
        - end_time_ms (int): End time in milliseconds.
        
        Returns:
        - torch.Tensor: Extracted and padded audio segment.
        """
        waveform, sr = torchaudio.load(audio_file_path)
        
        # Convert milliseconds to sample indices
        start_sample = int(start_time_ms * sr / 1000)
        end_sample = int(end_time_ms * sr / 1000)
        
        # Extract segment and resample if necessary
        segment = waveform[:, start_sample:end_sample]
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sample_rate)
            segment = resampler(segment)
        
        # Pad or truncate segment
        return self.pad_audio_segment(segment)
    
    def pad_audio_segment(self, segment):
        """
        Pads or truncates an audio segment to the target length.
        
        Parameters:
        - segment (torch.Tensor): The audio segment to be padded.
        
        Returns:
        - torch.Tensor: Padded or truncated audio segment.
        """
        current_length = segment.size(1)
        if current_length < self.target_length_samples:
            padding_size = self.target_length_samples - current_length
            padding = torch.full((segment.size(0), padding_size), self.padding_value)
            padded_segment = torch.cat((segment, padding), dim=1)
        else:
            padded_segment = segment[:, :self.target_length_samples]
        return padded_segment