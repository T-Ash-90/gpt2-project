# DialoGPT Chatbot

This project implements a conversational chatbot using the **DialoGPT-medium** model, a dialogue-optimized variant of GPT-2. It allows interactive chatting in the command-line interface, maintaining conversational context across multiple turns.

## Features

- **Interactive chat**: Type your message and the bot responds in real-time.
- **Turn-based memory**: Keeps track of recent conversation history to provide context-aware responses.
- **Dialogue-optimized model**: Uses `microsoft/DialoGPT-medium`, fine-tuned specifically for conversational interactions.
- **Controlled text generation**: Uses `top-k`, `top-p`, and `temperature` sampling to balance response coherence and diversity.
- **Token-efficient history**: Automatically manages conversation length to stay within model limits.

## Requirements

You’ll need the following Python libraries:

- `torch`: For running the language model.
- `transformers`: For loading DialoGPT and tokenizing input.

Install them using pip:

```bash
pip install torch transformers
```

## How it Works

### 🧠 Model

This chatbot uses the `microsoft/DialoGPT-medium` model, which is a version of GPT-2 fine-tuned specifically for dialogue. Unlike the base GPT-2, DialoGPT understands conversation context, leading to more realistic and appropriate responses.

### 💬 Chat Flow

1. **User Input**: You type a message into the terminal.
2. **Tokenization**: The message is tokenized and appended to the current chat history.
3. **Response Generation**: The model generates a reply based on the full conversation so far.
4. **Decoding**: The generated tokens are decoded into human-readable text.
5. **History Management**: Chat history is trimmed if it exceeds the token limit (to keep the context relevant and within the model’s 1024-token window).

### ⚙️ Generation Settings

The response generation uses sampling to create more human-like answers:

| Setting               | Description                                               | Value        |
|-----------------------|-----------------------------------------------------------|--------------|
| `temperature`         | Controls randomness. Lower = more focused.               | `0.7`        |
| `top_k`               | Considers only the top K token choices at each step.     | `50`         |
| `top_p`               | Nucleus sampling. Chooses from top tokens that sum to P. | `0.95`       |
| `no_repeat_ngram_size` | Prevents repetition of short phrases.                    | `3`          |

These parameters help balance creativity and coherence during generation.

### 🧹 History Trimming

To avoid running into model token limits:

- Only the most recent part of the conversation is kept.
- The full input (user + bot history) is clipped if it exceeds 1024 tokens.
- This keeps the model focused and prevents degradation of response quality.

## 📝 Notes

- This project runs completely **locally** — no API keys or internet connection are required after the model is downloaded.
- For improved response quality, you can upgrade to `DialoGPT-large` (requires more memory).
- If you prefer open-ended or creative generation (like storytelling), you can switch back to models like `gpt2`, `gpt2-medium`, or `gpt2-xl`.
- Responses are limited to recent context due to the model’s 1024-token maximum input length.
- DialoGPT is best suited for **short, multi-turn conversations**. It may not perform well in long or highly technical dialogues.

## 🙏 Credits

- **Model**: [`microsoft/DialoGPT-medium`](https://huggingface.co/microsoft/DialoGPT-medium)  
- **Libraries**: [Hugging Face Transformers](https://github.com/huggingface/transformers)  
- **Framework**: [PyTorch](https://pytorch.org/)  
- **Project**: Developed and modified for local, real-time conversation
