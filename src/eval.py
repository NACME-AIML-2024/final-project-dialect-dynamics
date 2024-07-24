import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from utils import create_file_mapping, process_transcripts, AudioDataset
from sklearn.model_selection import train_test_split
from model import WhisperProcessor, WhisperForConditionalGeneration
from tqdm import tqdm
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from datetime import datetime
from jiwer import wer
import argparse

# Parse arguments
parser = argparse.ArgumentParser(description='Evaluate ASR model')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for evaluation')
parser.add_argument('--subset', type=int, help='Subset size of the data')
args = parser.parse_args()

# Set device
device = torch.device("cpu")  # Change to "mps" for MacBook with Metal Performance Shaders

# Load and process data
root_dir = "/Users/romerocruzsa/Workspace/aiml/final-project-dialect-dynamics"
transcript_dir = os.path.join(root_dir, 'data/coraal/transcript/text/')
audio_dir = os.path.join(root_dir, 'data/coraal/audio/wav/')

paths_df = create_file_mapping(transcript_dir, audio_dir)
combined_transcript_df = process_transcripts(paths_df)
filtered_transcript_df = combined_transcript_df[~combined_transcript_df['Content'].str.contains(r'[\(\)\[\]/<>]')].reset_index(drop=True)

# Use subset if specified
if args.subset:
    data_subset = filtered_transcript_df.sample(args.subset)
else:
    data_subset = filtered_transcript_df

_, test_df = train_test_split(data_subset, test_size=0.2, random_state=42)

# Initialize processor and model
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en").to(device)

# Create dataset and dataloader
test_dataset = AudioDataset(test_df, transcript_dir, audio_dir, processor)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# Evaluation function
def test(model, test_loader, processor, device):
    model.eval()
    total_loss = 0
    true_transcriptions = []
    predicted_transcriptions = []

    with torch.no_grad():
        for batch in tqdm(test_loader, total=len(test_loader), desc='Testing'):
            inputs = batch['input_features'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_features=inputs, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            outputs = model.generate(input_features=inputs)
            transcriptions = processor.batch_decode(outputs, skip_special_tokens=True)
            true_transcriptions.extend(processor.batch_decode(labels, skip_special_tokens=True))
            predicted_transcriptions.extend(transcriptions)

    total_loss /= len(test_loader)
    predicted_transcriptions = [x.upper() for x in predicted_transcriptions]
    wer_score = wer(true_transcriptions, predicted_transcriptions)

    # Create a DataFrame with the actual and predicted transcriptions
    transcripts = pd.DataFrame({
        'actual': true_transcriptions,
        'predicted': predicted_transcriptions
    })

    return transcripts, total_loss, wer_score

# Perform evaluation
test_transcripts, test_loss, test_wer = test(model, test_loader, processor, device)
print(f"Running inference...\t Test Loss: {test_loss:.4f}, Word Error Rate (WER): {test_wer:.4f}")

# Visualize word clouds
def visualizing_wordcloud(transcripts):
    true_text = ' '.join(transcripts['actual'].str.lower())
    pred_text = ' '.join(transcripts['predicted'].str.lower())

    true_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(true_text)
    pred_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(pred_text)

    plt.figure(figsize=(10, 8))
    
    plt.subplot(2, 1, 1)
    plt.imshow(true_wordcloud, interpolation='bilinear')
    plt.title('Word Cloud for True Transcriptions')
    plt.axis('off')
    
    plt.subplot(2, 1, 2)
    plt.imshow(pred_wordcloud, interpolation='bilinear')
    plt.title('Word Cloud for Predicted Transcriptions')
    plt.axis('off')

    plt.show()

visualizing_wordcloud(test_transcripts)

# Save word cloud image
timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
wordcloud_filepath = os.path.join(root_dir, f"output/wordcloud/word-cloud_{timestamp}.png")
plt.savefig(wordcloud_filepath)
print(f"Word cloud image saved to {wordcloud_filepath}")
