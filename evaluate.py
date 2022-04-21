from util import load_processor
from transformers import Wav2Vec2ForCTC
import torch
import torchaudio
from vocab import restore


model_path = 'm3hrdadfi/wav2vec2-large-xlsr-persian-v3'
output_dir = './wav2vec2-nena'

processor = load_processor()
model = Wav2Vec2ForCTC.from_pretrained(model_path)
state_dict = torch.load('wav2vec2-nena/pytorch_model_11.bin', map_location='cpu')
model.load_state_dict(state_dict)

path = '20sec.wav'
waveform, sample_rate = torchaudio.load(path)
resample_rate = 16_000
resample = torchaudio.transforms.Resample(
    orig_freq=sample_rate,
    new_freq=resample_rate
)
waveform = resample(waveform)

features = processor(waveform[0], sampling_rate=16000, return_tensors='pt', padding=True).input_values

with torch.no_grad():
    logits = model(features).logits

pred_ids = torch.argmax(logits, dim=-1)

print(restore(processor.batch_decode(pred_ids)[0]))
