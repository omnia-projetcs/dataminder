import ollama

def summarize_text(text, model_name="ministral-3:8b"):
    if not text or not text.strip():
        return "No text provided for summarization."

    # Prompt definition
    prompt = f"""
Here is the content of a document. 

Your tasks are:
1. Detect the original language of the document and state it at the beginning.
2. Provide a detailed summary of this text in ENGLISH.
3. Translate all prose and descriptive text to ENGLISH.
4. CRITICAL: You must preserve any code blocks, scripts, or terminal commands EXACTLY as they appear in the original text. Do not translate code syntax, variable names, or commands.

Structure your response in Markdown format.

Document text:
---
{text}
---

Analysis and Summary (in English):
"""

    try:
        response = ollama.chat(model=model_name, messages=[
            {
                'role': 'user',
                'content': prompt
            }
        ])
        return response['message']['content']
    except Exception as e:
        print(f"Error communicating with Ollama: {e}")
        return f"Error during summarization: {e}\n\nCheck that the Ollama server is running and the model '{model_name}' is available."
