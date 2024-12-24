import streamlit as st
from openai import OpenAI
import json
import time
from typing import Optional, Dict, List, Any, Tuple
from config import Message, QualChecklist, AVAILABLE_MODELS, SYSTEM_PROMPT_TEMPLATE
from rag import (
    setup_pinecone,
    chunk_text,
    create_embeddings,
    upload_to_pinecone,
    get_relevant_context
    )


def initialize_session_state() -> None:
    """
    Initialize all necessary session state variables for the Streamlit application.
    
    This function sets up default values for:
        - Chat messages history
        - Qualification checklist and its last update timestamp
        - Selected AI model
        - System role description
        - Knowledge base loading status
    
    The session state persists across Streamlit reruns, ensuring conversation
    continuity and state management.
    """
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! How can I help you with your property search today?"}
        ]
    
    if "qual_checklist" not in st.session_state:
        st.session_state.qual_checklist = {
            "checklist": {},
            "last_update": time.time()
        }
        
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = AVAILABLE_MODELS[0]
        
    if "system_role" not in st.session_state:
        st.session_state.system_role = "You are an AI assistant that helps qualify real estate leads."
        
    if "knowledge_base_loaded" not in st.session_state:
        st.session_state.knowledge_base_loaded = False


def create_checklist_from_text(goals_text: str, openai_api_key: str
                               ) -> Dict[str, Optional[str]]:
    """
    Convert text goals into a structured checklist using OpenAI's API.
    
    Args:
        goals_text (str): Raw text containing qualification goals
        openai_api_key (str): OpenAI API key for making requests
    
    Returns:
        Dict[str, Optional[str]]: A dictionary where keys are goals and values are None
        
    Example:
        >>> goals = "Need to check: budget, timeline"
        >>> create_checklist_from_text(goals, api_key)
        {'budget': None, 'timeline': None}
    """
    client = OpenAI(api_key=openai_api_key)
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{
                "role": "user",
                "content": f"""
                Convert these qualification goals into a simple JSON structure where all values are null:
                {goals_text}
                
                Example format:
                {{
                    "budget": null,
                    "timeline": null
                }}
                """
            }]
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        st.error(f"Error creating checklist: {e}")
        return {}
    
    
def process_message(
    messages: List[Message],
    checklist: QualChecklist,
    openai_api_key: str,
    model: str,
    system_role: str,
    context: str = ""
    ) -> Tuple[Optional[str], Dict[str, Optional[str]]]:
    """
    Process a conversation message and update the qualification checklist.
    
    This function sends the conversation history, current checklist state, and any
    relevant context to the AI model and receives both a response message and
    updated checklist values.
    
    Args:
        messages (List[Message]): Conversation history
        checklist (QualChecklist): Current qualification checklist state
        openai_api_key (str): OpenAI API key
        model (str): Name of the OpenAI model to use
        system_role (str): System prompt defining the AI's role
        context (str, optional): Additional context from knowledge base
    
    Returns:
        Tuple[Optional[str], Dict[str, Optional[str]]]: 
            - The AI's response message
            - Updated checklist values
            
    Example:
        >>> msgs = [{"role": "user", "content": "My budget is $500,000"}]
        >>> checklist = {"checklist": {"budget": None}, "last_update": time.time()}
        >>> response, updated_checklist = process_message(msgs, checklist, api_key, "gpt-3.5-turbo", "You are an AI assistant")
        >>> print(updated_checklist)
        {'budget': '500000'}
    """
    client = OpenAI(api_key=openai_api_key)
    
    # Format complete system prompt
    formatted_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        system_role=system_role,
        goals_list=list(checklist["checklist"].keys()),
        current_checklist=checklist["checklist"],
        context=context if context else ""
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": formatted_prompt},
                *messages
            ],
            response_format={"type": "json_object"}
        )
        
        try:
            result = json.loads(response.choices[0].message.content)
            if "response" in result and "checklist" in result:
                # Filter checklist to only include our goals
                filtered_checklist = {
                    k: result["checklist"].get(k, None) 
                    for k in checklist["checklist"].keys()
                }
                return result["response"], filtered_checklist
            else:
                return response.choices[0].message.content, checklist["checklist"]
        except json.JSONDecodeError:
            return response.choices[0].message.content, checklist["checklist"]
            
    except Exception as e:
        st.error(f"Error processing message: {e}")
        return None, checklist["checklist"]


def create_streamlit_interface():
    """
    Create and configure the Streamlit user interface.
    
    This function sets up:
        - Main title and description
        - Sidebar with API key inputs
        - Model selection
        - Agent role configuration
        - Knowledge base upload
        - Qualification goals selection
        - Real-time qualification status display
    """
    st.title("🤖 💬 AI Real Estate Lead Qualification Agent")
    st.caption("🚀 Powered by OpenAI & Pinecone")

    with st.sidebar:
        openai_api_key = st.text_input(
            "OpenAI API Key", 
            key="openai_api_key", 
            type="password"
            )
        "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)" 

        selected_model = st.selectbox(
            "Select Model",
            AVAILABLE_MODELS,
            index=0,
            key="selected_model" 
            )

        system_role = st.text_area(
            "Agent Role Description",
            value="You are an AI assistant that helps qualify real estate leads.",
            height=100,
            key="system_role"
            )
           
        pinecone_api_key = st.text_input(
            "Pinecone API Key", 
            key="pinecone_api_key", 
            type="password")
        "[Get an Pinecone API Key](https://app.pinecone.io/organizations/-/keys)"

        knowledge_file = st.file_uploader(
            "Upload Knowledge Base",
            type="txt",
            help="Upload a TXT file with your knowledge base"
        )
        if knowledge_file and not st.session_state.knowledge_base_loaded:
            if not st.session_state.get("pinecone_api_key"):
                st.error("Please add your Pinecone API key first to upload knowledge base.")
            else:
                text_content = knowledge_file.getvalue().decode()
                
                with st.spinner("Processing and uploading knowledge base..."):
                    chunks = chunk_text(text_content)
                    openai_client = OpenAI(api_key=st.session_state.openai_api_key)
                    embeddings = create_embeddings(chunks, openai_client)
                    index = setup_pinecone(st.session_state.pinecone_api_key)
                    
                    if index and upload_to_pinecone(index, chunks, embeddings):
                        st.session_state.knowledge_base_loaded = True
                        st.success("Knowledge base uploaded successfully!")

        st.subheader("Qualification Goals")

        AVAILABLE_GOALS = [
            "Budget",
            "Timeline",
            "Location Preference",
            "Property Type",
            "Number of Bedrooms",
            "Investment or Living",
            "Contact Phone",
            "Email Address",
            "Preferred Viewing Time"
        ]

        selected_goals = st.multiselect(
            "Select qualification goals",
            AVAILABLE_GOALS,
            default=["Email Address", "Budget"],
            key="selected_goals"
            )

        if selected_goals:
            current_goals = set(st.session_state.qual_checklist["checklist"].keys())
            selected_goals_set = set(selected_goals)
    
            if current_goals != selected_goals_set:
                st.session_state.qual_checklist = {
                    "checklist": {goal: None for goal in selected_goals},
                    "last_update": time.time()
                    }

        if "qual_checklist" in st.session_state:
            st.subheader("Qualification Status")
            st.json(st.session_state.qual_checklist["checklist"])
            
        
 
def main():
    """
    Main application function that orchestrates the chat interface and message processing.
    
    This function:
        1. Initializes the session state
        2. Creates the Streamlit interface
        3. Manages the chat message history
        4. Processes user inputs
        5. Integrates with knowledge base when available
        6. Updates and displays the qualification checklist
        
    The function runs in a continuous loop, handling user interactions and
    updating the interface in real-time through Streamlit's reactive framework.
    """
    initialize_session_state()
    create_streamlit_interface()
    
    # Display message history
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
    
    # Handle user input
    if prompt := st.chat_input("Type your message here..."):
        if not st.session_state.openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()
        
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        # Get context if Pinecone is set up
        context = ""
        if (st.session_state.get("pinecone_api_key") and 
            st.session_state.get("knowledge_base_loaded")):
            with st.spinner("Searching relevant information..."):
                index = setup_pinecone(st.session_state.pinecone_api_key)
                if index:
                    context = get_relevant_context(
                        prompt,
                        index,
                        OpenAI(api_key=st.session_state.openai_api_key)
                        )
        
        # Process message
        with st.spinner("Thinking..."):
            response, updated_checklist = process_message(
                st.session_state.messages,
                st.session_state.qual_checklist,
                st.session_state.openai_api_key,
                st.session_state.selected_model,
                st.session_state.system_role,
                context
            )
        
            if response:
                # Update checklist and add assistant's response
                st.session_state.qual_checklist["checklist"] = updated_checklist
                st.session_state.qual_checklist["last_update"] = time.time()
            
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.chat_message("assistant").write(response)

                st.rerun()

if __name__ == "__main__":
    main()