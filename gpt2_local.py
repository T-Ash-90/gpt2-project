import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GenerationConfig

# Load the GPT-2 model and tokenizer from Hugging Face
model_name = "gpt2-medium"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Set the model to evaluation mode
model.eval()

# Maximum token length for input and output
MAX_HISTORY_LENGTH = 500  # Max length of conversation history to maintain

# Function to generate text based on a prompt
def generate_text(prompt, max_length=500):
    # Encode the prompt to token IDs
    inputs = tokenizer.encode(prompt, return_tensors="pt")

    # Create attention mask (a tensor of ones)
    attention_mask = torch.ones(inputs.shape, dtype=torch.long)  # Create a mask of ones

    # Generation configuration (using `do_sample` for randomness control)
    outputs = model.generate(
        inputs,
        max_length=min(max_length + len(inputs[0]), 1024),  # Limit total token length to 1024 tokens
        num_return_sequences=1,  # Number of sequences to generate
        no_repeat_ngram_size=3,  # Increase n-gram repetition penalty
        pad_token_id=tokenizer.eos_token_id,  # Explicitly set pad_token_id to eos_token_id
        eos_token_id=tokenizer.eos_token_id,  # Stop when EOS token is generated
        attention_mask=attention_mask,  # Explicit attention mask
        do_sample=True,  # Enable sampling for more natural responses
        top_k=50,  # Top-k sampling for diversity
        top_p=0.9,  # Top-p (nucleus) sampling to limit randomness
        temperature=0.7,  # Lower temperature for more predictable answers
    )

    # Decode the generated tokens back to text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove the user input from the response (if it exists at the start)
    if generated_text.lower().startswith(prompt.lower()):
        generated_text = generated_text[len(prompt):].strip()

    return generated_text

# Function to maintain conversation history
def chat():
    conversation_history = ""  # Start with an empty history
    print("Chat with GPT-2! Type 'exit' to end.")

    while True:
        user_input = input("\033[96mYou: \033[0m")  # User input in cyan color

        if user_input.lower() == "exit":
            break

        # Add the user input to the conversation history
        conversation_history += f"Human: {user_input}\nBot: "

        # Trim conversation history to avoid exceeding max token length
        tokenized_history = tokenizer.encode(conversation_history)
        if len(tokenized_history) > MAX_HISTORY_LENGTH:
            conversation_history = tokenizer.decode(tokenized_history[-MAX_HISTORY_LENGTH:])

        # Generate the response
        response = generate_text(conversation_history, max_length=500)

        # Add GPT-2 response to the conversation history
        conversation_history += f"{response}\n"

        # Print the GPT-2 response in yellow
        print(f"\033[93mBot: {response}\033[0m")  # \033[93m is for yellow color

# Start the interactive chat
chat()
