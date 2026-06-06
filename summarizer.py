import re
from llm_client import LLMClient

# Pattern to strip conversational preamble (e.g. "Okay, here's a breakdown...")
_PREAMBLE_RE = re.compile(
    r'\A\s*(?:okay|alright|sure|here(?:\'s| is)|let me|great|certainly|of course|i\'ll)[^\n]*\n+',
    re.IGNORECASE
)

# Pattern to strip trailing non-knowledge sections
_TRAILING_SECTIONS_RE = re.compile(
    r'\n+(?:#{1,3}\s*|\*{2})(?:'
    r'resources|references|sources|bibliography|links|contact'
    r'|further (?:reading|learning)|next steps|to help me'
    r'|additional (?:resources|notes|considerations)'
    r'|in summary|summary|conclusion|key takeaways'
    r')(?:\*{2})?[:\s].*',
    re.DOTALL | re.IGNORECASE
)

# Pattern to strip trailing conversational questions
_TRAILING_QUESTIONS_RE = re.compile(
    r'\n+\*{2}(?:to help me|would you like|do you want|let me know|if you)[^\n]*\*{2}[:\s].*',
    re.DOTALL | re.IGNORECASE
)


def _clean_output(text):
    """Remove conversational preamble, trailing resource sections, and questions."""
    text = _PREAMBLE_RE.sub('', text)
    text = _TRAILING_SECTIONS_RE.sub('', text)
    text = _TRAILING_QUESTIONS_RE.sub('', text)
    return text.strip()


def summarize_text(text, model_name="gemma3:4b-it-q4_K_M", level=7, llm_client=None):
    if not text or not text.strip():
        return "No text provided for summarization."

    if llm_client is None:
        llm_client = LLMClient(provider="ollama")

    # Adjust instruction based on level
    if level < 1:
        level = 1
    elif level > 10:
        level = 10
        
    length_instruction = f"The desired detail level is {level} out of 10. "
    if level <= 2:
        length_instruction += "Make it extremely brief and concise. Only the most critical information."
    elif level <= 5:
        length_instruction += "Provide a balanced summary covering main points."
    elif level <= 8:
        length_instruction += "Provide a detailed and thorough extraction. Include all important technical details, examples, and specifics."
    else:
        length_instruction += "Be MAXIMALLY exhaustive. Extract EVERY piece of knowledge, every detail, every example, every command, every configuration. Do not omit anything. The output should be as long and complete as possible."

    system_prompt = (
        "You are a silent technical knowledge extractor. You are NOT a chatbot. "
        "You NEVER talk to the user. You NEVER ask questions. You NEVER add commentary. "
        "You NEVER review or describe the document itself. "
        "You output ONLY structured Markdown containing extracted technical knowledge. "
        "No preamble, no introduction, no conclusion, no suggestions, no questions. "
        "Start directly with the first technical heading."
    )

    # Prompt definition
    prompt = f"""{length_instruction}

RULES:
1. Write everything in ENGLISH regardless of the source language.
2. Extract ONLY real knowledge: technical facts, definitions, commands, tools, techniques, configurations, protocols, procedures, code examples, specific values, parameters, attack methods, defense strategies, etc.
3. Preserve code blocks, scripts, commands, and terminal output EXACTLY as they appear. Do not translate code.
4. Write the knowledge as standalone facts, as if you are a technical instructor teaching the subject.
5. Start IMMEDIATELY with a Markdown heading. No introductory text, no preamble, no "here is...".
6. End with the last piece of technical knowledge. Nothing after it.
7. PROGRESSIVE EXTRACTION: Process the text sequentially from start to finish. Extract the actual technical elements one by one in detail. DO NOT provide a high-level summary of what the document is (e.g., do not say "This is a massive list of tools" or "Overall Observations"). Detail the technical elements themselves.

FORBIDDEN — Do NOT include any of the following:
- "Overall Observations", "General Summary", or any commentary on the document's nature, age, or quality.
- Document metadata: title, author, publisher, ISBN, edition, publication date
- Structural descriptions: "Purpose and Philosophy", "Target Audience", "Document Structure", "Learning Objectives", "Prerequisites", "Overview of the book"
- Meta-references: "this book explains...", "the author describes...", "in chapter 3...", "according to the document...", "this list contains..."
- Generic filler: "security is important", "cybersecurity is a growing field"
- Resources/References/Sources sections: URLs, links, contact information, email addresses, phone numbers, social media, "for more information visit...", recommended readings, bibliographies
- Next steps, further learning, suggestions, advice, or recommendations
- Questions to the user such as "would you like...", "to help me tailor...", "do you want..."
- Any conversational text or commentary

Structure the output in Markdown using technical topic headings.

Document:
---
{text}
---
"""

    try:
        content = llm_client.chat(
            model=model_name,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt}
            ],
            keep_alive=-1
        )
        return _clean_output(content)
    except Exception as e:
        print(f"Error communicating with LLM ({llm_client}): {e}")
        return f"Error during summarization: {e}\n\nCheck that your LLM server is running and the model '{model_name}' is available."
