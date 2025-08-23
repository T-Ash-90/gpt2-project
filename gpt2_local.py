# gpt2_local.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "microsoft/DialoGPT-medium"
MAX_HISTORY_TOKENS = 1000  # Max tokens to keep in chat history
MAX_RESPONSE_TOKENS = 50   # Max tokens generated per reply

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.eval()

def is_math_question(text):
    ops = ['+', '-', '*', 'x', 'times', '/']
    return any(op in text for op in ops)

def solve_math_question(text):
    import re
    try:
        expression = text.lower().replace('times', '*').replace('x', '*')
        # Extract numbers and operators simply
        numbers = re.findall(r'[\d\.]+', expression)
        operators = re.findall(r'[\+\-\*\/]', expression)
        if len(numbers) == 2 and len(operators) == 1:
            expr = f"{numbers[0]}{operators[0]}{numbers[1]}"
            answer = eval(expr)
            # Format answer nicely: int if whole number
            if answer == int(answer):
                answer = int(answer)
            return str(answer)
    except:
        pass
    return None

def polite_fallback():
    fallback_replies = [
        "I'm sorry, could you please rephrase that?",
        "I want to help, but I didn't quite understand. Could you try again?",
        "That's interesting! Could you tell me more?",
        "I appreciate your question. Let me think about it.",
        "I'm here to assist you, please ask me anything."
    ]
    import random
    return random.choice(fallback_replies)

def main():
    print("Chat with DialoGPT! Type 'exit' to end.")

    # Start with a system prompt to set personality
    system_prompt = (
        "The following is a polite and helpful conversation between a human and an AI assistant. "
        "The AI is friendly, informative, and always responds politely.\n"
    )
    chat_history_ids = tokenizer.encode(system_prompt, return_tensors="pt")

    while True:
        user_input = input("\033[96mYou:\033[0m ").strip()
        if user_input.lower() == "exit":
            print("Goodbye! Have a great day!")
            break

        # Handle math questions directly
        if is_math_question(user_input):
            answer = solve_math_question(user_input)
            if answer:
                print(f"\033[93mAI:\033[0m The answer is {answer}.")
                # Update history with user and bot messages
                user_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
                answer_ids = tokenizer.encode(answer + tokenizer.eos_token, return_tensors='pt')
                chat_history_ids = torch.cat([chat_history_ids, user_ids, answer_ids], dim=-1)
                # Trim history tokens
                if chat_history_ids.size(-1) > MAX_HISTORY_TOKENS:
                    chat_history_ids = chat_history_ids[:, -MAX_HISTORY_TOKENS:]
                continue

        # Otherwise, generate response with the model
        new_user_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
        input_ids = torch.cat([chat_history_ids, new_user_ids], dim=-1)

        # Attention mask: all ones since no padding here
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long)

        chat_history_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=input_ids.shape[-1] + MAX_RESPONSE_TOKENS,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            no_repeat_ngram_size=3,
            eos_token_id=tokenizer.eos_token_id,
        )

        # Decode only the new tokens
        response = tokenizer.decode(chat_history_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True).strip()

        # Ensure polite fallback if response empty or nonsensical
        if not response:
            response = polite_fallback()

        print(f"\033[93mAI:\033[0m {response}")

        # Update history with the new interaction
        # Concatenate user input + model response (both with eos tokens)
        user_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
        response_ids = tokenizer.encode(response + tokenizer.eos_token, return_tensors='pt')
        chat_history_ids = torch.cat([chat_history_ids, user_ids, response_ids], dim=-1)

        # Trim conversation history to avoid exceeding max tokens
        if chat_history_ids.size(-1) > MAX_HISTORY_TOKENS:
            chat_history_ids = chat_history_ids[:, -MAX_HISTORY_TOKENS:]

if __name__ == "__main__":
    main()
