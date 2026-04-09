import shutil, os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

blobs = r'C:\Users\CHINNI PRASANNA\.cache\huggingface\hub\models--Qwen--Qwen2-0.5B-Instruct\blobs'
snap = r'C:\Users\CHINNI PRASANNA\.cache\huggingface\hub\models--Qwen--Qwen2-0.5B-Instruct\snapshots\c540970f9e29518b1d8f06ab8b24cba66ad77b6d'

files = {
    '463b055262b6c66c4629a74a4b300bfe2ed31d3c': 'config.json',
    'ff55d7b9eb1384e5d4d7e75dc0f564c1a8833d6e': 'tokenizer_config.json',
    '33ea6c72ebb92a237fa2bdf26c5ff16592efcdae': 'tokenizer.json',
    '4783fe10ac3adce15ac8f358ef5462739852c569': 'vocab.json',
    '20024bfe7c83998e9aeaf98a0cd6a2ce6306c2f0': 'merges.txt',
    'dfc11073787daf1b0f9c0f1499487ab5f4c93738': 'generation_config.json',
    '130282af0dfa9fe5840737cc49a0d339d06075f83c5a315c3372c9a0740d0b96': 'model.safetensors',
}

print("Copying files to snapshot folder...")
for blob, name in files.items():
    src = os.path.join(blobs, blob)
    dst = os.path.join(snap, name)
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.copy2(src, dst)
        print(f'Copied: {name}')

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(snap)
model = AutoModelForCausalLM.from_pretrained(snap, torch_dtype=torch.float32)

messages = [
    {"role": "system", "content": "You are a helpful weather safety assistant."},
    {"role": "user", "content": "Is it safe to go outside in heavy rain?"}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7, do_sample=True)

reply = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print("\n🤖 Chatbot says:", reply)