"""
    llama-4b trainer with support of Stanford Alpaca-like JSON datasets (short for SAD)
    Intended to use with https://github.com/johnsmith0031/alpaca_lora_4bit

    SAD structure:
    [
        {
            "instruction": "Give null hypothesis",
            "input": "6 subjects were given a drug (treatment group) and an additional 6 subjects a placebo (control group).",
            "output": "Drug is equivalent of placebo"
        },
        {
            "instruction": "What does RNA stand for?",
            "input": "",
            "output": "RNA stands for ribonucleic acid."
        }
    ]
"""

import sys
import tqdm

import evaluate as evaluate
import peft
import peft.tuners.lora
from torch.nn import CrossEntropyLoss
from transformers import get_linear_schedule_with_warmup

assert peft.tuners.lora.is_gptq_available()

import bitsandbytes as bnb
import torch
import transformers

from accelerate import Accelerator
from autograd_4bit import load_llama_model_4bit_low_ram
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, PeftModel, set_peft_model_state_dict
from torch import nn
from torch.utils.data.dataloader import DataLoader
from transformers.trainer_pt_utils import get_parameter_names

# ! Config
from arg_parser import get_config
import train_data

accelerator = Accelerator()
device = accelerator.device

ft_config = get_config()

# * Show loaded parameters
if ft_config.local_rank == 0:
    print(f"{ft_config}\n")

if ft_config.gradient_checkpointing:
    print('Disable Dropout.')

# Load Basic Model
model, tokenizer = load_llama_model_4bit_low_ram(ft_config.llama_q4_config_dir,
                                                  ft_config.llama_q4_model,
                                                  device_map=ft_config.device_map,
                                                  groupsize=ft_config.groupsize)

# Config Lora
lora_config = LoraConfig(
    r=ft_config.lora_r,
    lora_alpha=ft_config.lora_alpha,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=ft_config.lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
)
if ft_config.lora_apply_dir is None:
    model = get_peft_model(model, lora_config)
else:
    model = PeftModel.from_pretrained(model, ft_config.lora_apply_dir, device_map={'': 0}, torch_dtype=torch.float32)  # ! Direct copy from inference.py
    print(ft_config.lora_apply_dir, 'loaded')


# Scales to half
print('Fitting 4bit scales and zeros to half')
for n, m in model.named_modules():
    if '4bit' in str(type(m)):
        if m.groupsize == -1:
            m.zeros = m.zeros.half()
        m.scales = m.scales.half()

# Set tokenizer
tokenizer.pad_token_id = 0

if not ft_config.skip:
    # Load Data
    data = None
    if ft_config.ds_type == "txt" and not ft_config.skip:
        #### LLaMa
        data = train_data.TrainTxt(ft_config.dataset, ft_config.val_set_size, tokenizer, ft_config.cutoff_len)
    elif ft_config.ds_type == "alpaca" and not ft_config.skip:
        #### Stanford Alpaca-like Data
        data = train_data.TrainSAD(ft_config.dataset, ft_config.val_set_size, tokenizer, ft_config.cutoff_len)
    elif ft_config.ds_type == "gpt4all" and not ft_config.skip:
        #### GPT4All Data
        data = train_data.TrainGPT4All(ft_config.dataset, ft_config.val_set_size, tokenizer, ft_config.cutoff_len)
    else:
        raise NotImplementedError("ERROR: Unknown dataset format")
    data.prepare_data(thd=ft_config.txt_row_thd, use_eos_token=ft_config.use_eos_token)
    ####

    # Use gradient checkpointing
    if ft_config.gradient_checkpointing:
        print('Applying gradient checkpointing ...')
        from gradient_checkpointing import apply_gradient_checkpointing
        apply_gradient_checkpointing(model, checkpoint_ratio=ft_config.gradient_checkpointing_ratio)

    training_args = transformers.TrainingArguments(
        per_device_train_batch_size=ft_config.mbatch_size,
        gradient_accumulation_steps=ft_config.gradient_accumulation_steps,
        warmup_steps=ft_config.warmup_steps,
        num_train_epochs=ft_config.epochs,
        learning_rate=ft_config.lr,
        fp16=True,
        logging_steps=ft_config.logging_steps,
        evaluation_strategy="no",
        save_strategy="steps",
        eval_steps=None,
        save_steps=ft_config.save_steps,
        output_dir=ft_config.lora_out_dir,
        save_total_limit=ft_config.save_total_limit,
        load_best_model_at_end=False,
        ddp_find_unused_parameters=False if ft_config.ddp else None,
    )

    trainer_kwargs = {}

    decay_parameters = get_parameter_names(model, [nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]
    optimizer_kwargs = {
        "betas": (training_args.adam_beta1, training_args.adam_beta2),
        "eps": training_args.adam_epsilon,
    }
    optimizer_kwargs["lr"] = training_args.learning_rate
    adam_bnb_optim = bnb.optim.Adam8bit(
        optimizer_grouped_parameters,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        eps=training_args.adam_epsilon,
        lr=training_args.learning_rate,
    )
    trainer_kwargs['optimizers'] = (adam_bnb_optim, None)

    def custom_collate(batch):
        max_length = max([len(item["input_ids"]) for item in batch])
        padded_input_ids = []
        padded_attention_mask = []

        for item in batch:
            input_ids = item["input_ids"]
            attention_mask = item["attention_mask"]
            padding_length = max_length - len(input_ids)

            padded_input_ids.append(input_ids + [tokenizer.pad_token_id] * padding_length)
            padded_attention_mask.append(attention_mask + [0] * padding_length)

        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
        }

    train_dataloader = DataLoader(data.train_data, batch_size=ft_config.mbatch_size, collate_fn=custom_collate)
    val_dataloader = DataLoader(data.val_data, batch_size=ft_config.mbatch_size, collate_fn=custom_collate)

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=adam_bnb_optim,
        num_warmup_steps=training_args.warmup_steps,
        num_training_steps=(len(train_dataloader) * training_args.num_train_epochs),
    )

    metric = evaluate.load("accuracy", "neg_log_loss")
    if ft_config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Set Model dict
    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
    ).__get__(model, type(model))

    model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(model, adam_bnb_optim, train_dataloader, val_dataloader, lr_scheduler)

    # Set Verbose
    if ft_config.verbose:
        transformers.logging.set_verbosity_info()

    loss_function = CrossEntropyLoss()

    if ft_config.resume_checkpoint:
        print('Resuming from {} ...'.format(ft_config.resume_checkpoint))
        adapters_weights = torch.load(ft_config.resume_checkpoint)
        model = set_peft_model_state_dict(model, adapters_weights)

    model.train()
    for epoch in range(int(training_args.num_train_epochs)):
        for step, batch in enumerate(t:=tqdm.tqdm(train_dataloader)):
            batch = {k: v.to(accelerator.device) for k, v in batch.items()}
            with accelerator.accumulate(model):
                output = model(**batch)
                logits = output.logits
                loss = loss_function(logits.view(-1, logits.shape[-1]), batch["input_ids"].view(-1))
                loss = loss.half()
                t.set_description(f"step loss: {loss.cpu().float()}")
                accelerator.backward(loss)
                adam_bnb_optim.step()
                lr_scheduler.step()
                adam_bnb_optim.zero_grad()
        model.eval()
        for step, batch in enumerate(val_dataloader):
            batch = {k: v.to(accelerator.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            predictions, references = accelerator.gather_for_metrics((predictions, batch["input_ids"]))
            metric.add_batch(
                predictions=predictions,
                references=references,
            )

        eval_metric = metric.compute()
        # Use accelerator.print to print only on the main process.
        accelerator.print(f"epoch {epoch}:", eval_metric)

    print('Train completed.')

# Save Model
model.save_pretrained(ft_config.lora_out_dir)

if ft_config.checkpoint:
    print("Warning: Merge model + LoRA and save the whole checkpoint not implemented yet.")

print('Model Saved.')
