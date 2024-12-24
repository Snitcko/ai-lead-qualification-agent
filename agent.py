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
    """Initialize all necessary session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! How can I help you with your property search today?"}
        ]
    
    if "qual_checklist" not in st.session_state:
        st.session_state.qual_checklist = {
            "checklist": {},
            "last_update": time.time()
        }
        
    # Добавляем инициализацию остальных переменных
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = AVAILABLE_MODELS[0]
        
    if "system_role" not in st.session_state:
        st.session_state.system_role = "You are an AI assistant that helps qualify real estate leads."
        
    if "knowledge_base_loaded" not in st.session_state:
        st.session_state.knowledge_base_loaded = False

def create_checklist_from_text(goals_text: str, openai_api_key: str) -> Dict[str, Optional[str]]:
    """Convert text goals into a structured checklist."""
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
    """Process message and update checklist."""
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



# Настраиваем заголовок и описание приложения
def create_streamlit_interface():
    """Create the Streamlit user interface."""
    st.title("🤖 💬 AI Real Estate Lead Qualification Agent")
    st.caption("🚀 Powered by OpenAI & Pinecone")

# Настраиваем боковую панель
    with st.sidebar:
        # API Keys
        openai_api_key = st.text_input(
            "OpenAI API Key", 
            key="openai_api_key", 
            type="password"
            )
        # Ссылка на получение API ключа
        "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)" 
        # Model Selection
        selected_model = st.selectbox(
            "Select Model",
            AVAILABLE_MODELS,
            index=0, # По умолчанию выбрана первая модель
            key="selected_model" 
            )
        # Agent Role
        system_role = st.text_area(
            "Agent Role Description",
            value="You are an AI assistant that helps qualify real estate leads.",
            height=100,
            key="system_role"
            )
           
        pinecone_api_key = st.text_input(
            "Pinecone API Key", 
            key="pinecone_api_key", 
            type="password") # Скрываем ключ звездочками
        # Ссылка на получение API ключа
        "[Get an Pinecone API Key](https://app.pinecone.io/organizations/-/keys)"
        
        # Knowledge Base Upload
        knowledge_file = st.file_uploader(
            "Upload Knowledge Base",
            type="txt",
            help="Upload a TXT file with your knowledge base"
        )
            
        # Обрабатываем только если файл новый и еще не загружен
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
        
        
        
        
        
        # Новый интерфейс для добавления целей
        st.subheader("Qualification Goals")
        
        # Предустановленные цели
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
        
        # Мультиселект для выбора целей
        selected_goals = st.multiselect(
            "Select qualification goals",
            AVAILABLE_GOALS,
            default=["Email Address", "Budget"],
            key="selected_goals"
        )
        
        # Создаем или обновляем чеклист на основе выбранных целей
        if selected_goals:
            # Обновляем только если изменился список целей
            current_goals = set(st.session_state.qual_checklist["checklist"].keys())
            selected_goals_set = set(selected_goals)
    
            if current_goals != selected_goals_set:
                st.session_state.qual_checklist = {
                    "checklist": {goal: None for goal in selected_goals},
                    "last_update": time.time()
                    }

        # Показываем текущий статус квалификации всегда
        if "qual_checklist" in st.session_state:
            st.subheader("Qualification Status")
            st.json(st.session_state.qual_checklist["checklist"])
            
        
        
        
def main():
    """Main application function."""
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
        
#         # Добавим отображение статуса квалификации в боковой панели
#         st.divider() # Разделительная линия
#         st.subheader("Qualification Status")
#     #     #st.json(st.session_state.qualification_status) # Отображаем статус в JSON формате



#     # # используем session_state для сохранения данных между обновлениями
#     if "messages" not in st.session_state:
#         st.session_state["messages"] = [
#             {"role": "assistant", "content": "How can I help you?"}
#             ]
#     # if "qualification_status" not in st.session_state:
#     #     st.session_state["qualification_status"] = {
#     #         "status": "pending",
#     #         "message": {},
#     #     }

#     # Отображение истории сообщений (Создаем UI элементы для каждого сообщения с соответствующими стилями)
#     for msg in st.session_state.messages:
#         st.chat_message(msg["role"]).write(msg["content"])
        
#     # Обработка ввода пользователя
#     # walrus оператор, который присваивает и возвращает значение
#     if prompt := st.chat_input("Say something"): # walrus оператор, который присваивает и возвращает значение
#         if not openai_api_key:  # Проверяем наличие API ключа
#             st.info("Please add your OpenAI API key to continue.")
#             st.stop()
#         # # Для отладки
#         # st.write("Debug - Current prompt:", prompt)
#         # st.write("Debug - Current messages:", st.session_state.messages)

#         # Добавляем сообщение пользователя в историю и отображаем его
#         st.session_state.messages.append({"role": "user", "content": prompt})
#         st.chat_message("user").write(prompt)


# # AI
# # Функция для получения ответа от OpenAI
# def get_openai_response(messages: List[Dict], openai_api_key: str, model: str) -> str:
#     client = OpenAI(api_key=openai_api_key) # Создаем клиент OpenAI с предоставленным API ключом
#     try:
#         # Создаем запрос к API OpenAI и добавляем системное сообщение с историей сообщений
#         response = client.chat.completions.create(
#             model=model,
#             messages=[
#                 {"role": "system", "content": SYSTEM_MESSAGE},
#                 *messages # Добавляем историю сообщений
#                 ]
#             )
#         return response.choices[0].message.content # Возвращаем ответ от OpenAI
#     except Exception as e: # В случае ошибки выводим сообщение
#         st.error(f"An error occurred: {e}")
#         return None

# # Получаем ответ от OpenAI и показываем спиннер во время загрузки
# with st.spinner("Thinking..."):
#     response = get_openai_response(
#         st.session_state.messages, 
#         openai_api_key, 
#         selected_model
#         )       
# # Если ответ получен, добавляем его в историю и отображаем
# if response:
#     st.session_state.messages.append({"role": "assistant", "content": response})
#     st.chat_message("assistant").write(response)
#     #st.session_state.qualification_status["status"] = "complete" # Устанавливаем статус квалификации (todo: добавить логику квалификации)
        

