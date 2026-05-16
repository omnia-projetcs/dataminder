import ollama

def summarize_text(text, model_name="ministral-3:8b"):
    if not text or not text.strip():
        return "No text provided for summarization."

    # Prompt definition
    prompt = f"""
Here is the content of a document. Provide a detailed summary of this text, highlighting the most important points. 
Structure your response in Markdown format (with headings, bullet points for key points, etc.).

Document text:
---
{text}
---

Detailed summary and key points:
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
