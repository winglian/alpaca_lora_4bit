import os
import sys
import time
import torch
import autograd_4bit

from autograd_4bit import load_llama_model_4bit_low_ram, Autograd4bitQuantLinear
from peft import PeftModel
from peft.tuners.lora import Linear4bitLt
from torch import nn

config_path = '../models/decapoda-research_llama-13b-hf/'
model_path = '../models/decapoda-research_llama-13b-hf-int4/llama-13b-4bit.pt'
lora_path = './alpaca_lora/'

model, tokenizer = load_llama_model_4bit_low_ram(config_path, model_path, groupsize=-1)
model = PeftModel.from_pretrained(model, lora_path, device_map={'': 0}, torch_dtype=torch.float32)

print('Fitting 4bit scales and zeros to half')
# model.half()
for n, m in model.named_modules():
    if isinstance(m, Autograd4bitQuantLinear) or isinstance(m, Linear4bitLt):
        if isinstance(m, Autograd4bitQuantLinear):
            if m.groupsize == -1:
                m.zeros = m.zeros.half()
            m.scales = m.scales.half()
            try:
                m.bias = m.bias.half()
            except TypeError:
                m.bias = nn.Parameter(m.bias.half())
autograd_4bit.use_new = True
autograd_4bit.auto_switch = True

print('Apply AMP Wrapper ...')
from amp_wrapper import AMPWrapper
wrapper = AMPWrapper(model)
wrapper.apply_generate()

prompt_template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""
instruction = '''Write a python function to recursively add the individual digits of a number together until a single digit is reached. For example if provided with the value 259, add the digits 2+5+9 to get 16, then add 1+6 to get 7 as the answer.\n'''
input = ""
batch = tokenizer(prompt_template.format(instruction=instruction, input=input), return_tensors="pt", add_special_tokens=False)
batch = {k: v.cuda() for k, v in batch.items()}

start = time.time()
with torch.no_grad():
    generated = model.generate(inputs=batch["input_ids"],
                               do_sample=True, use_cache=True,
                               repetition_penalty=1.1,
                               max_new_tokens=20,
                               temperature=0.9,
                               top_p=0.95,
                               top_k=40,
                               return_dict_in_generate=True,
                               output_attentions=False,
                               output_hidden_states=False,
                               output_scores=False)
result_text = tokenizer.decode(generated['sequences'].cpu().tolist()[0])
end = time.time()
print(result_text)
print(end - start)
