# Lead Qualification AI Agent 🏠 🤖

An intelligent chatbot powered by OpenAI's GPT and Pinecone vector database that helps qualify real estate leads through natural conversation. The agent collects important information about potential buyers while maintaining a friendly and professional dialogue.

## Features

- 🎯 Customizable qualification goals (budget, timeline, preferences, etc.)
- 💬 Natural conversation flow with context awareness
- 📚 Knowledge base integration through RAG (Retrieval-Augmented Generation)
- 📊 Real-time qualification status tracking
- 🔄 Seamless integration with OpenAI's latest models
- 📱 Clean and responsive Streamlit interface

## Architecture

The project consists of three main components:

1. **Agent Module** (`agent.py`)
   - Streamlit interface implementation
   - Chat message processing
   - Session state management
   - Qualification checklist handling

2. **Configuration Module** (`config.py`)
   - System prompt templates
   - Model configurations
   - Type definitions
   - Pinecone settings

3. **RAG Module** (`rag.py`)
   - Knowledge base processing
   - Text chunking and embedding
   - Vector database operations
   - Context retrieval

## Prerequisites

- Python 3.8+
- OpenAI API key
- Pinecone API key (for knowledge base features)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Snitcko/ai-lead-qualification-agent.git
cd real-estate-ai-agent
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

1. Obtain API keys:
   - Get an OpenAI API key from [OpenAI Platform](https://platform.openai.com/account/api-keys)
   - Get a Pinecone API key from [Pinecone Console](https://app.pinecone.io/organizations/-/keys)

2. The API keys can be entered directly in the Streamlit interface.

## Usage

1. Start the application:
```bash
streamlit run agent.py
```

2. Open the provided URL in your browser (typically `http://localhost:8501`)

3. Enter your API keys in the sidebar

4. Select qualification goals you want to track

5. Optionally upload a knowledge base TXT file for enhanced responses

6. Start chatting with the agent!

## Customization

### Qualification Goals

You can modify the available qualification goals in `agent.py`:

```python
AVAILABLE_GOALS = [
    "Budget",
    "Timeline",
    "Location Preference",
    # Add your custom goals here
]
```

### System Prompt

Modify the agent's behavior by adjusting the system prompt template in `config.py`.

### Knowledge Base

Prepare your knowledge base as a text file with relevant real estate information, guidelines, or FAQs. The system will automatically chunk and embed this information for contextual responses.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- OpenAI for their powerful language models
- Pinecone for vector similarity search capabilities
- Streamlit for the amazing web interface framework

## Support

If you find this project helpful, please give it a ⭐ on GitHub!