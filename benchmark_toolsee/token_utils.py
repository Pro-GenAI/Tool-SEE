from typing import Any, Dict, List, Tuple

import tiktoken
_encoding = tiktoken.get_encoding("o200k_harmony")

# from transformers import AutoTokenizer
# _transformers_tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")


def count_tokens(text: Any) -> int:
	"""Return number of tokens for given input using tiktoken if available,
	otherwise fall back to a Transformers tokenizer.
	"""
	return len(_encoding.encode(str(text)))
	# toks = _transformers_tokenizer.encode(str(text), add_special_tokens=False)
	# return len(toks)


def count_tokens_for_tool_list(tools: List[Tuple[str, Dict[str, Any]]]) -> int:
	total_tokens = 0
	for _, tool_data in tools:
		total_tokens += count_tokens(tool_data)
	return total_tokens


if __name__ == "__main__":
	sample_text = "Hello, world! This is a test string to count tokens."
	print(f"Token count: {count_tokens(sample_text)}")
