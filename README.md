# GPT-2 Chatbot

This project implements a conversational chatbot using the GPT-2 language model. It allows interactive chatting with the bot in a command-line interface. The bot generates responses based on user input, maintaining the conversation history for context.

## Features
- **Interactive chat**: Type your messages and the bot will respond.
- **Token history management**: The conversation history is kept within a manageable length to avoid exceeding the model's token limit.
- **Sampling-based generation**: The bot uses sampling techniques (`top-k`, `top-p`, and `temperature`) to make the responses more diverse and natural.
- **Customizable token length**: You can modify the maximum length of conversation history and generated responses.

## Requirements
To run this project, you'll need the following Python libraries:
- `torch`: For running the GPT-2 model.
- `transformers`: For loading the GPT-2 model and tokenizer from Hugging Face.

You can install the dependencies using `pip`:

```bash
pip install torch transformers
```

## How it Works

### Model Loading

The GPT-2 model and tokenizer are loaded using the Hugging Face `transformers` library. The model used in this example is `gpt2-medium`, but you can replace it with other GPT-2 variants (like `gpt2-xl` or `gpt2-large`) depending on the available resources. Larger models require more memory.

### Generating Responses

1. The input text (your messages) is tokenized and passed into the GPT-2 model.
2. The bot generates a response by predicting the next set of tokens (words/phrases) based on the input.
3. The response is decoded back into human-readable text.

### Conversation History

To ensure that the bot provides contextually relevant responses, the conversation history is stored and passed along with each new input. The history is limited to a maximum number of tokens to prevent it from exceeding the model’s capacity (1024 tokens).
