from megatron.tokenizer.tokenizer import build_tokenizer
from argparse import ArgumentParser, Namespace
from megatron.tokenizer.tokenizer import build_tokenizer

def main(args: Namespace) -> None:
    """Main: create C4/pile streaming dataset.

    Args:
        args (Namespace): Commandline arguments.
    """
    # Write samples
    args.rank = 0
    args.model_parallel_size = 1
    args.make_vocab_size_divisible_by = 128
    tokenizer = build_tokenizer(args)
    tokenizer.tokenizer.add_special_tokens([f"<|p|>"])
    tokenizer.tokenizer.add_special_tokens([f"<|/p|>"])
    tokenizer.tokenizer.add_special_tokens([f"<|box|>"])
    tokenizer.tokenizer.add_special_tokens([f"<|/box|>"])
    tokenizer.tokenizer.add_special_tokens([f"<|grounding|>"])
    tokenizer.tokenizer.add_special_tokens([f"<|image_start|>"])
    tokenizer.tokenizer.add_special_tokens([f"<|image_end|>"])

    for i in range(1024):
        tokenizer.tokenizer.add_special_tokens([f"<|box_{i}|>"])

    for i in range(8192):
        tokenizer.tokenizer.add_special_tokens([f"<|seed_{i}|>"])

    tokenizer.tokenizer.save(
        "/p/project/ccstdl/gupta6/multimodal/20B_tokenizer_final.json"
    )


def parse_args() -> Namespace:
    """Parse commandline arguments."""
    parser = ArgumentParser(
        description="Add new tokens to the vocabulary of a tokenizer."
    )
    parser.add_argument("--tokenizer_type", type=str, required=False, default=None)
    parser.add_argument("--vocab_file", type=str, required=False, default=None)
    parser.add_argument("--merge_file", type=str, required=False, default=None)

    parsed = parser.parse_args()
    return parsed


if __name__ == "__main__":
    main(parse_args())
