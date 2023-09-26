from transformers.models.robin import RobinConfig, RobinForCausalLM
from transformers import PreTrainedTokenizerFast
import torch

ROBIN_HF = 'checkpoints/robin_hf/'
device = torch.device("cuda")
tokenizer = PreTrainedTokenizerFast.from_pretrained(ROBIN_HF)
print('Tokenizer loaded')

model = RobinForCausalLM.from_pretrained(ROBIN_HF).half().to(device)
model.eval()
print('Model loaded')

# Normal forward as text
prompt = "Hello, my name is"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
outputs = model.generate(
    inputs=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    do_sample=True,
    temperature=0.1,
    max_length=100,
)
text = tokenizer.batch_decode(outputs)[0]

print(text)

    
