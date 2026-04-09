import torch
import shutil
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

# Model path
BLOBS = r'C:\Users\CHINNI PRASANNA\.cache\huggingface\hub\models--Qwen--Qwen2-0.5B-Instruct\blobs'
SNAP = r'C:\Users\CHINNI PRASANNA\.cache\huggingface\hub\models--Qwen--Qwen2-0.5B-Instruct\snapshots\c540970f9e29518b1d8f06ab8b24cba66ad77b6d'

FILES = {
    '463b055262b6c66c4629a74a4b300bfe2ed31d3c': 'config.json',
    'ff55d7b9eb1384e5d4d7e75dc0f564c1a8833d6e': 'tokenizer_config.json',
    '33ea6c72ebb92a237fa2bdf26c5ff16592efcdae': 'tokenizer.json',
    '4783fe10ac3adce15ac8f358ef5462739852c569': 'vocab.json',
    '20024bfe7c83998e9aeaf98a0cd6a2ce6306c2f0': 'merges.txt',
    'dfc11073787daf1b0f9c0f1499487ab5f4c93738': 'generation_config.json',
    '130282af0dfa9fe5840737cc49a0d339d06075f83c5a315c3372c9a0740d0b96': 'model.safetensors',
}

def _ensure_files():
    for blob, name in FILES.items():
        src = os.path.join(BLOBS, blob)
        dst = os.path.join(SNAP, name)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)

# System personality for your weather project
SYSTEM_PROMPT = """You are a helpful weather and disaster risk assistant. 
You help people understand weather conditions, safety tips, and disaster risks. 
Keep answers short, clear and helpful."""

class WeatherChatbot:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.loaded = False

    def load(self):
        """Load model into memory — call this once at startup"""
        if not self.loaded:
            print("Loading chatbot model... please wait")
            _ensure_files()
            self.tokenizer = AutoTokenizer.from_pretrained(SNAP)
            self.model = AutoModelForCausalLM.from_pretrained(
                SNAP, torch_dtype=torch.float32
            )
            self.loaded = True
            print("Chatbot ready!")

    def ask(self, user_question, weather_context=""):
        """Ask the chatbot a question"""
        if not self.loaded:
            self.load()

        # Add weather context if provided
        system = SYSTEM_PROMPT
        if weather_context:
            system += f"\n\nCurrent weather info: {weather_context}"

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_question}
        ]

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt")

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True
            )

        reply = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        return reply.strip()


# Single instance to reuse across your app
chatbot = WeatherChatbot()