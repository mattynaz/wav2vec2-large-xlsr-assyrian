import os
import json
from datasets import load_dataset, load_metric, concatenate_datasets
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2CTCTokenizer
from datacollator import DataCollatorCTCWithPadding
from vocab import vocab, normalize
import torch
import torchaudio
from typing import Dict
from audiomentations import *
import numpy as np
from copy import deepcopy



def load_processor(
    vocab: Dict[str, int] = vocab,
    unk_token: str = '[UNK]',
    pad_token: str = '[PAD]',
    word_delimiter_token: str = ' ',
    temp_path: str = '__vocab__.json'
) -> Wav2Vec2Processor:
    '''
    `vocab`: jhjkh
    '''

    if os.path.exists(temp_path):
        raise Exception(
            f'Cannot use "{temp_path}" as temp path.'
            'File already exists here.'
        )

    vocab = vocab.copy()
    vocab[unk_token] = len(vocab)
    vocab[pad_token] = len(vocab)
    with open(temp_path, 'w') as f:
        json.dump(vocab, f)

    tokenizer = Wav2Vec2CTCTokenizer(
        temp_path,
        unk_token=unk_token,
        pad_token=pad_token,
        word_delimiter_token=word_delimiter_token,
        eos_token=None,
        bos_token=None,
    )

    os.remove(temp_path)

    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=False,
    )

    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
    )

    return processor


def load_data_collator(processor):
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    return data_collator


def load_nena_dataset(processor, data_files='nena_dataset.json', augment=True, duplicate_dataset=1, test_split=0.075):
    # Initialize dataset
    dataset = load_dataset('json', data_files=data_files)
    if duplicate_dataset > 1:
        copies = [deepcopy(dataset['train']) for _ in range(duplicate_dataset)]
        dataset['train'] = concatenate_datasets(copies)

    augments = Compose([
        Gain(min_gain_in_db=-12, max_gain_in_db=6, p=0.75),
        PitchShift(min_semitones=-6, max_semitones=1, p=0.75),
        TimeStretch(min_rate=0.70, max_rate=1.20, p=0.75),
        RoomSimulator(p=0.25),
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.008, p=0.75),
    ])

    # Prepare and split dataset
    def prepare(item):
        waveform, sample_rate = torchaudio.load(item['path'])

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

    dataset = dataset.map(
        prepare,
        remove_columns=dataset.column_names['train'],
        num_proc=8,
    )

    if test_split > 0:
        dataset = dataset['train'].train_test_split(test_size=test_split)

    return dataset


wer_metric = load_metric('wer')
cer_metric = load_metric('cer')

def compute_metrics(pred, processor):
        pred_logits = torch.as_tensor(pred.predictions)
        pred_ids = torch.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        cer = cer_metric.compute(predictions=pred_str, references=label_str)

        return {'wer': wer, 'cer': cer}
