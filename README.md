# 🤖 my-finetuned-model

A fine-tuned **OpenChat 3.5** model customized using a domain-specific **JSONL dataset** and deployed with an **ASR → LLM → TTS** conversational pipeline using **Gradio**.

---

## 🌐 Live Demo on Hugging Face

[![Hugging Face](https://img.shields.io/badge/HuggingFace-Model-yellow?logo=huggingface)](https://huggingface.co/arsheefathimas/my-finetuned-openchat)  
👉 [Click here to view and download the model](https://huggingface.co/arsheefathimas/my-finetuned-openchat)

---

## 🚀 Key Highlights

- 🧠 Fine-tuned version of OpenChat 3.5  
- 🗣️ Converts voice to text using **Whisper Tiny**  
- 💬 Generates context-aware replies using the fine-tuned model  
- 🔊 Responds with voice using **gTTS (Google Text-to-Speech)**  
- 🧩 Built with Gradio and runs inside Colab (A100 GPU used)

---

## 🧪 Base Model

- **Base:** [OpenChat 3.5](https://huggingface.co/openchat/openchat_3.5)  
- **Architecture:** Mistral  
- **Params:** 7.24B  
- **Precision:** FP16  
- **Chat Format:** ChatML / OpenChat-style templating

---

## 🛠️ Training Setup

- **Data format:** JSONL with role-based dialogue  
- **Fine-tuning:** Used LLaMA Factory + Flash Attention 2 on Colab A100  
- **Epochs:** 3  
- **Batch Size:** 1  
- **Tokenizer:** Inherited from base OpenChat model

---

## 🔗 Repositories

- 💾 **Model:** [Hugging Face - arsheefathimas/my-finetuned-openchat](https://huggingface.co/arsheefathimas/my-finetuned-openchat)  
- 💻 **Gradio App:** [Hugging Face Space - Yalnaverse Chat](https://huggingface.co/spaces/arsheefathimas/yalnaverse_chat)  
- 📦 **Source Code (this repo):** [`my-finetuned-model`](https://github.com/ArsheeFathimaS/my-finetuned-model)

---

## 🪪 License

> 📄 Apache 2.0 — Same as the base OpenChat model.

---

## 💡 How to Use

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

tokenizer = AutoTokenizer.from_pretrained("arsheefathimas/my-finetuned-openchat")
model = AutoModelForCausalLM.from_pretrained("arsheefathimas/my-finetuned-openchat")

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
prompt = "I'm unable to pass level 21. Can you help?"
print(generator(prompt, max_new_tokens=100)[0]["generated_text"])

