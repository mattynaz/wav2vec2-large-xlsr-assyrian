{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "from util import load_processor\n",
    "from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC\n",
    "import torch\n",
    "import torchaudio\n",
    "from vocab import restore\n",
    "\n",
    "model_path = 'm3hrdadfi/wav2vec2-large-xlsr-persian-v3'\n",
    "output_dir = './wav2vec2-nena'\n",
    "\n",
    "processor = load_processor()\n",
    "model = Wav2Vec2ForCTC.from_pretrained(model_path)\n",
    "state_dict = torch.load('wav2vec2-nena/pytorch_model.bin', map_location='cpu')\n",
    "model.load_state_dict(state_dict)\n",
    "\n",
    "path = 'mom_01.wav'\n",
    "waveform, sample_rate = torchaudio.load(path)\n",
    "resample_rate = 16_000\n",
    "resample = torchaudio.transforms.Resample(\n",
    "        orig_freq=sample_rate,\n",
    "        new_freq=resample_rate\n",
    "    )\n",
    "waveform = resample(waveform)\n",
    "\n",
    "features = processor(waveform[0], sampling_rate=16_000, return_tensors='pt', padding=True).input_values\n",
    "\n",
    "with torch.no_grad():\n",
    "    logits = model(features).logits\n",
    "\n",
    "pred_ids = torch.argmax(logits, dim=-1)\n",
    "\n",
    "print(restore(processor.batch_decode(pred_ids)[0]))\n",
    "\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Users/matthew/Desktop/nena/asr/venv/lib/python3.9/site-packages/transformers/configuration_utils.py:358: UserWarning: Passing `gradient_checkpointing` to a config initialization is deprecated and will be removed in v5 Transformers. Using `model.gradient_checkpointing_enable()` instead, or if you are using the `Trainer` API, pass `gradient_checkpointing=True` in your `TrainingArguments`.\n",
      "  warnings.warn(\n"
     ]
    }
   ],
   "source": [
    "model_path = 'm3hrdadfi/wav2vec2-large-xlsr-persian-v3'\n",
    "output_dir = './wav2vec2-nena'\n",
    "\n",
    "model = Wav2Vec2ForCTC.from_pretrained(model_path)\n",
    "\n",
    "# processor = Wav2Vec2Processor.from_pretrained('wav2vec2-nena/preprocessor_config.json')\n",
    "# model = Wav2Vec2ForCTC.from_pretrained('wav2vec2-nena/config.json')\n",
    "state_dict = torch.load('wav2vec2-nena/pytorch_model.bin', map_location='cpu')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<All keys matched successfully>"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.load_state_dict(state_dict)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "i u ʾa hva p r xa p ul h i +x l ya +ra ba +p ya y ya ha j u x\n"
     ]
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 720x144 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "path = 'mom_01.wav'\n",
    "waveform, sample_rate = torchaudio.load(path)\n",
    "resample_rate = 16_000\n",
    "resample = torchaudio.transforms.Resample(\n",
    "        orig_freq=sample_rate,\n",
    "        new_freq=resample_rate\n",
    "    )\n",
    "waveform = resample(waveform)\n",
    "\n",
    "processor = load_processor()\n",
    "features = processor(waveform[0], sampling_rate=16_000, return_tensors='pt', padding=True).input_values\n",
    "\n",
    "with torch.no_grad():\n",
    "    logits = model(features).logits\n",
    "\n",
    "pred_ids = torch.argmax(logits, dim=-1)\n",
    "\n",
    "print(restore(processor.batch_decode(pred_ids)[0]))\n"
   ]
  }
 ],
 "metadata": {
  "interpreter": {
   "hash": "5d40c2c6fb4bd461f0e8df8b274491dc63b2e0b0ab8000f94d08640890039470"
  },
  "kernelspec": {
   "display_name": "Python 3.9.12 ('venv': venv)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.12"
  },
  "orig_nbformat": 4
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
