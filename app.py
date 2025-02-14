import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Cargar modelo y tokenizador
model_name = "microsoft/DialoGPT-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

st.title("🤖 Chatbot de Recetas")

# Historial de conversación
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Entrada del usuario
user_input = st.text_input("Escribe tu pregunta sobre recetas:", "")

if user_input:
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
    bot_input_ids = torch.cat([torch.tensor(st.session_state.chat_history), new_input_ids], dim=-1) if st.session_state.chat_history else new_input_ids
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

    st.session_state.chat_history = chat_history_ids.tolist()
    st.write(f"🤖 Chatbot: {response}")
