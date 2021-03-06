from functools import partial
import numpy as np
from vocab import normalize
import torch
import torchaudio
from audiomentations import Compose, Gain, PitchShift, TimeStretch, RoomSimulator, AddGaussianNoise
from datacollator import DataCollatorCTCWithPadding


augments = Compose([
    Gain(min_gain_in_db=-2, max_gain_in_db=2, p=0.25),
    PitchShift(min_semitones=-6, max_semitones=1, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.1, p=0.25),
    # RoomSimulator(p=0.15),
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.006, p=1),
])


def prepare(item, processor, augment):
    sample_rate = item['sample_rate']
    waveform = torch.as_tensor(item['waveform'])

    if augment:
        waveform = augments(np.array(waveform), sample_rate=sample_rate)
        waveform = torch.as_tensor(waveform)

    resample_rate = 16000
    resample = torchaudio.transforms.Resample(
        orig_freq=sample_rate,
        new_freq=resample_rate
    )
    waveform = resample(waveform)

    item['input_values'] = processor(
        waveform[0],
        sampling_rate=resample_rate
    ).input_values[0]

    with processor.as_target_processor():
        text = normalize(item['utterance'])
        item['labels'] = processor(text).input_ids

    return item


def prepare_dataset(dataset, processor, augment=True, num_proc=16):
    dataset['train'] = dataset['train'].map(
        partial(prepare, processor=processor, augment=augment),
        remove_columns=dataset.column_names['train'],
        num_proc=num_proc,
    )
    dataset['test'] = dataset['test'].map(
        partial(prepare, processor=processor, augment=False),
        remove_columns=dataset.column_names['test'],
        num_proc=num_proc,
    )
    return dataset

def load_data_collator(processor):
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    return data_collator
