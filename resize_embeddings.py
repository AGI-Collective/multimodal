from transformers import GPTNeoXForCausalLM, AutoTokenizer
from megatron.tokenizer.tokenizer import build_tokenizer


model = GPTNeoXForCausalLM.from_pretrained(
  "EleutherAI/pythia-1.4b-deduped-v0",
  revision="step143000",
  cache_dir="/p/fastdata/mmlaion/hummingbird/checkpoints/pythia1_4",
)

tokenizer = AutoTokenizer.from_pretrained(
  "EleutherAI/pythia-1.4b-deduped-v0",
  revision="step143000",
  cache_dir="/p/fastdata/mmlaion/hummingbird/checkpoints/pythia1_4",
)

tokenizer_args = {
        "tokenizer_type": "HFTokenizer",
        "vocab_file": "/p/project/ccstdl/gupta6/multimodal/20B_tokenizer.json",
        "rank": 0,
        "model_parallel_size": 1,
        "make_vocab_size_divisible_by": 128,
    }
class Config:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)


tokenizer_args = Config(tokenizer_args)
# tokenizer_config = om.create(tokenizer_args)
tokenizer_new = build_tokenizer(tokenizer_args)
new_vocab_size = tokenizer_new.vocab_size
print("new_vocab_size", new_vocab_size)
print("old_vocab_size", tokenizer.vocab_size)

model.resize_token_embeddings(new_vocab_size, pad_to_multiple_of=128)

model.save_pretrained("/p/fastdata/mmlaion/hummingbird/checkpoints/pythia1_4_resized")
inputs = tokenizer("Hello, I am", return_tensors="pt")
tokens = model.generate(**inputs)
print(tokenizer.decode(tokens[0]))

