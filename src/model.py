from transformers import WhisperProcessor, WhisperForConditionalGeneration, WhisperConfig

# Define the WhisperProcessor and WhisperForConditionalGeneration
def get_whisper_model():
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
    config = WhisperConfig.from_pretrained("openai/whisper-tiny.en")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
    return processor, model
