import torch
from functools import partial
from transformers import Wav2Vec2ForCTC, TrainingArguments, Trainer
from util import *

processor = load_processor()
data_collator = load_data_collator(processor)
dataset = load_nena_dataset(processor, duplicate_dataset=1, augment=True)
compute_metrics = partial(compute_metrics, processor=processor)

model_path = 'm3hrdadfi/wav2vec2-large-xlsr-persian-v3'
output_dir = './wav2vec2-nena'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(model_path, device)

model = Wav2Vec2ForCTC.from_pretrained(model_path).to(device)

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

model.freeze_feature_extractor()
model.gradient_checkpointing_enable()

training_args = TrainingArguments(
      output_dir='./wav2vec2-nena',
      group_by_length=True,
      per_device_train_batch_size=12,
      gradient_accumulation_steps=4,
      evaluation_strategy='steps',
      num_train_epochs=75,
      fp16=True,
      save_steps=1000,
      eval_steps=5,
      logging_steps=5,
      learning_rate=4e-3,
      warmup_ratio=0.20,
      save_total_limit=2,
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

trainer.train()

trainer.save_model()