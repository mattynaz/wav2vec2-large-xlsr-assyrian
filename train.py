import torch
from functools import partial
from transformers import Wav2Vec2ForCTC, TrainingArguments, Trainer
from util import *
import sys
from datautils import prepare_dataset, load_data_collator


model_path = 'm3hrdadfi/wav2vec2-large-xlsr-persian-v3'
model_version = 'main'
augment = False

if len(sys.argv) == 2:
    model_path = sys.argv[1]
if len(sys.argv) == 3:
    model_version = sys.argv[2]
if len(sys.argv) == 4:
    augment = 'augment' == sys.argv[3]


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Training "{model_path}@{model_version}" on "{device}".')
if augment:
    print('Data augmentation on.')
print('\n')

if 'mnazarix' in model_path:
    processor = Wav2Vec2Processor.from_pretrained(model_path, revision=model_version)
else:
    processor = load_processor()

model = Wav2Vec2ForCTC.from_pretrained(
    model_path, 
    attention_dropout=0.1,
    hidden_dropout=0.1,
    feat_proj_dropout=0.0,
    mask_time_prob=0.05,
    layerdrop=0.1,
    ctc_loss_reduction='mean', 
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer)
)
model.gradient_checkpointing_enable()

dataset = load_dataset('mnazari/urmi-assyrian-voice')
dataset = prepare_dataset(dataset, processor, augment=augment, num_proc=16)
data_collator = load_data_collator(processor)
compute_metrics = partial(compute_metrics, processor=processor)

training_args = TrainingArguments(
      output_dir='./output',
      group_by_length=True,
      per_device_train_batch_size=8,
      gradient_accumulation_steps=2,
      evaluation_strategy='steps',
      num_train_epochs=1,
      fp16=True,
      eval_steps=50,
      logging_steps=10,
      learning_rate=3e-3,
      warmup_ratio=0.2,
  )

trainer = Trainer(
      model=model,
      data_collator=data_collator,
      args=training_args,
      compute_metrics=compute_metrics,
      train_dataset=dataset['train'],
      eval_dataset=dataset['test'],
      tokenizer=processor.feature_extractor,
  )

# trainer.train()

hub_token = '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'
trainer.push_to_hub('mnazari/delete_this_later', token=hub_token)
