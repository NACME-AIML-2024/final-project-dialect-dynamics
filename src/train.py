import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from utils import create_file_mapping, process_transcripts, AudioDataset
from model import WhisperProcessor, WhisperForConditionalGeneration, WhisperConfig
from tqdm import tqdm
from jiwer import wer
from datetime import datetime

# Set device
device = torch.device("cpu")  # Change to "mps" for MacBook with Metal Performance Shaders

# Load and process data
root_dir = "/Users/romerocruzsa/Workspace/aiml/final-project-dialect-dynamics"
transcript_dir = os.path.join(root_dir, 'data/coraal/transcript/text/')
audio_dir = os.path.join(root_dir, 'data/coraal/audio/wav/')

paths_df = create_file_mapping(transcript_dir, audio_dir)
combined_transcript_df = process_transcripts(paths_df)
filtered_transcript_df = combined_transcript_df[~combined_transcript_df['Content'].str.contains(r'[\(\)\[\]/<>]')].reset_index(drop=True)
data_subset = filtered_transcript_df.sample(150)
train_df, test_df = train_test_split(data_subset, test_size=0.2, random_state=42)

# Initialize processor and model
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
config = WhisperConfig.from_pretrained("openai/whisper-tiny.en")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en").to(device)

# Create datasets and dataloaders
train_dataset = AudioDataset(train_df, transcript_dir, audio_dir, processor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
# import pdb;pdb.set_trace()

# Training function
def train(model, train_loader, processor, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    true_transcriptions = []
    predicted_transcriptions = []

    for batch in tqdm(train_loader, total=len(train_loader), desc='Training'):
        inputs = batch['input_features'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_features=inputs, labels=labels)
        loss = outputs.loss

        outputs = model.generate(input_features=inputs)
        transcriptions = processor.batch_decode(outputs, skip_special_tokens=True)
        true_transcriptions.extend(processor.batch_decode(labels, skip_special_tokens=True))
        predicted_transcriptions.extend(transcriptions)

        predicted_transcriptions = [x.upper() for x in predicted_transcriptions]
        wer_score = wer(true_transcriptions, predicted_transcriptions)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
    total_loss /= len(train_loader)

    return total_loss, wer_score

# Setup optimizer and scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
num_epochs = 1
train_loss_per_epoch = []
train_wer_per_epoch = []

# Training loop
for epoch in range(num_epochs):
    train_loss, train_wer = train(model, train_loader, processor, optimizer, scheduler, device)
    print(f"Epoch {epoch+1}/{num_epochs}\t Training Loss: {train_loss:.4f}, Word Error Rate (WER): {train_wer:.4f}")
    train_loss_per_epoch.append(train_loss)
    train_wer_per_epoch.append(train_wer)

# Save training metrics to CSV
metrics_df = pd.DataFrame({
    'Epoch': range(1, num_epochs + 1),
    'Training Loss': train_loss_per_epoch,
    'Word Error Rate (WER)': train_wer_per_epoch
})

# Save model metrics and weights
timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
metrics_df.to_csv(os.path.join(root_dir, f"output/metrics/training-metrics_{timestamp}.csv"), index=False)
weights_filepath = os.path.join(root_dir, f"output/weights/whisper-weights_{timestamp}")
torch.save(model.state_dict(), weights_filepath)
print(f"Model weights saved to {weights_filepath}")
