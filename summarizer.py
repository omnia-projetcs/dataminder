import ollama

def summarize_text(text, model_name="ministral-3:8b", level=7):
    if not text or not text.strip():
        return "No text provided for summarization."

    # Adjust instruction based on level
    if level < 1:
        level = 1
    elif level > 10:
        level = 10
        
    length_instruction = f"The desired detail level for the summary is {level} out of 10. "
    if level <= 3:
        length_instruction += "Make it extremely brief, concise, and straight to the point. Only the most critical information."
    elif level <= 7:
        length_instruction += "Provide a balanced, detailed summary covering all main points."
    else:
        length_instruction += "Make it highly exhaustive and comprehensive. Do not leave out any details, provide an extensive summary."

    # Prompt definition
    prompt = f"""
Here is the content of a document. 

Your tasks are:
1. Detect the original language of the document and state it at the beginning.
2. Provide a summary of this text in ENGLISH. {length_instruction}
3. Translate all prose and descriptive text to ENGLISH.
4. CRITICAL: You must preserve any code blocks, scripts, or terminal commands EXACTLY as they appear in the original text. Do not translate code syntax, variable names, or commands.
5. CRITICAL: NEVER reference the source document, book, author, chapter, or publication in your summary. Do NOT write things like "this book explains...", "the author describes...", "in chapter 3...", "according to the document...". Write the knowledge directly as standalone facts, as if you are teaching the subject yourself.

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
