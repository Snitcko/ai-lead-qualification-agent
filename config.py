from typing import TypedDict, Dict, Optional, Any

class Message(TypedDict):
    role: str
    content: str

class QualChecklist(TypedDict):
    checklist: Dict[str, Optional[str]]
    last_update: float

# OpenAI Configuration
AVAILABLE_MODELS = [
    "gpt-3.5-turbo-0125",
    "gpt-4o-mini-2024-07-18",
    "gpt-4o-2024-08-06",
    ]

# Pinecone Configuration
PINECONE_DIMENSION = 1536  # Dimension for OpenAI ada-002 embeddings
PINECONE_METRIC = "cosine"
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"
CHUNK_SIZE = 600  # Size of text chunks for knowledge base
CHUNK_OVERLAP = 100  # Overlap between chunks

# System prompts
SYSTEM_PROMPT_TEMPLATE = """{{system_role}}

You must collect information about these topics: {goals_list}
Current status of collected information: {current_checklist}

Guidelines:
1. Respond in the same language the user is using
2. Focus on collecting one piece of information at a time
3. Integrate questions naturally into the conversation
4. When you receive information, store the exact value in the checklist
5. Ask follow-up questions if the provided information is unclear
6. If user declines to provide information, store "declined" in the checklist

Example checklist updates:
- If user says "My email is john@example.com", store exactly "john@example.com"
- If user says "My budget is 500,000", store "500,000"
- If user says "I don't want to share my email", store "declined"

{context}

Remember to return your response ONLY in JSON format with exact values:
{{
    "response": "your message in user's language",
    "checklist": {{
        "goal1": "exact value or null or declined"
    }}
}}"""