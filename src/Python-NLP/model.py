from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load pretrained GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

def generate_reponse(prompt: str) -> str:
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    # Generate text based on the prompt
    output = model.generate(input_ids, max_length=50, no_repeat_ngram_size=12)
    # Decode and print the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text
