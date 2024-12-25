import os
import gradio as gr
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pathlib import Path
from typing import Annotated, List, Dict, Literal, Union
import requests
import time
import logging
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# Load environment variables
load_dotenv()

# Define the state schema
class State(TypedDict):
    messages: Annotated[list, add_messages]
    cards: List[Dict]
    current_card_index: int
    card_data: List[Dict]
    message_type: str  # 'card_list' or 'question'

def initialize_chain():
    """
    Initialize the LangChain chat model and chain
    """
    chat_model = ChatOpenAI(
        temperature=0.3,
        model_name="gpt-4o-mini",
        streaming=True
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant. Provide clear and concise responses."),
        ("human", "{message}")
    ])
    
    chain = prompt | chat_model | StrOutputParser()
    return chain

def read_file_content(file_path: str) -> str:
    """
    Read the content of the uploaded text file
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return f"Here is the list of cards (the last one is the commander):\n\n{content}"
    except Exception as e:
        return f"Error reading file: {str(e)}"

def process_uploaded_file(file_obj) -> str:
    """
    Process the uploaded file and return its content
    """
    if file_obj is None:
        return "No file uploaded"
    return read_file_content(file_obj.name)

def determine_message_type(state: State) -> Union[Literal["process_cards"], Literal["answer_question"]]:
    """
    Determine whether to process cards or answer a question directly.
    """
    print(f"Determining message type... Message type is: {state['message_type']}")
    return "process_cards" if state["message_type"] == "card_list" else "answer_question"

def answer_question(state: State) -> Dict:
    """
    Answer a question using existing card data if available.
    """
    return {
        "messages": ["Answering question directly without fetching new card data"],
        "card_data": state.get("card_data", [])
    }

def create_card_processing_graph() -> StateGraph:
    """
    Create the graph for processing cards with conditional routing.
    """
    print("\n=== Creating card processing graph ===")
    # Initialize graph builder
    graph = StateGraph(State)
    
    print("Adding nodes to graph...")
    # Add nodes
    graph.add_node("process_card", process_card)
    graph.add_node("answer_question", answer_question)
    
    # Add start node that passes through state
    def start(state: State) -> State:
        print(f"Start node called with state: {state}")
        return state
    
    graph.add_node("start", start)
    
    def end(state: State) -> State:
        print(f"End node called with state: {state}")
        return state
    
    graph.add_node("end", end)
    
    print("Adding conditional edges...")
    # Add conditional edges
    graph.add_conditional_edges(
        "process_card",
        should_continue,
        {
            "process_card": "process_card",
            "end": "end"
        }
    )
    
    graph.add_conditional_edges(
        "start",
        determine_message_type,
        {
            "process_cards": "process_card",
            "answer_question": "answer_question"
        }
    )
    
    print("Setting entry point to 'start'")
    graph.set_entry_point("start")
    
    print("Compiling graph...")
    return graph.compile()

def chat_response(message: str, history: list) -> str:
    """
    Generate chat response using card data.
    """
    print("\n=== Starting new chat response ===")
    print(f"Message: {message}")
    print(f"History exists: {bool(history)}")
    
    # Initialize chain
    chain = initialize_chain()
    
    # Determine if message is a card list by checking for card list markers
    is_card_list = ("list of cards" in message.lower() or 
                   any(line.strip().startswith(("1 ", "2 ")) for line in message.split('\n')))
    message_type = "card_list" if is_card_list else "question"
    print(f"Determined message type: {message_type}")
    
    # Parse cards if it's a card list
    if message_type == "card_list":
        # Clear any existing card data
        if hasattr(chat_response, 'card_data'):
            print("Clearing previous session's card data")
            delattr(chat_response, 'card_data')
        
        # Parse cards from the message
        cards = [line.strip() for line in message.split('\n') 
                if line.strip() and (line.strip().startswith("1 ") or line.strip().startswith("2 "))]
        cards = [card[2:].strip() for card in cards]  # Remove the "1 " prefix
        print(f"Parsed cards: {cards if cards else 'No cards parsed'}")
    else:
        cards = []
    
    # Create and run the graph
    graph = create_card_processing_graph()
    
    # Initial state
    initial_state = {
        "messages": [],
        "cards": cards,
        "current_card_index": 0,
        "card_data": getattr(chat_response, 'card_data', []),
        "message_type": message_type
    }
    print(f"Initial state: {initial_state}")
    
    # Process through graph
    print("Starting graph processing...")
    final_state = graph.invoke(initial_state, {"recursion_limit": 200})
    print(f"Graph processing complete. Final state: {final_state}")
    
    # Store card data if new data was fetched
    if message_type == "card_list" and final_state.get("card_data"):
        chat_response.card_data = final_state["card_data"]
        print(f"Stored {len(final_state['card_data'])} cards in memory")
    
    # Prepare context for the response
    if hasattr(chat_response, 'card_data') and chat_response.card_data:
        context = "Available card data:\n"
        for card in chat_response.card_data:
            context += f"- {card['name']}: {card.get('mana_cost', 'N/A')} | {card.get('type_line', 'N/A')} | {card.get('oracle_text', 'N/A')}\n"
        augmented_message = f"{context}\nUser question: {message}"
    else:
        augmented_message = message
    print(f"Prepared message for LLM: {augmented_message[:200]}...")
    
    # Generate response
    print("Generating response...")
    response = ""
    for chunk in chain.stream({"message": augmented_message}):
        response += chunk
        yield response

@tool
def fetch_scryfall_data(card_name: str) -> Dict:
    """
    Fetches card data from Scryfall API for a single card.
    
    Args:
        card_name: Name of the Magic card to look up
        
    Returns:
        Dictionary containing card information including name, mana cost, type, etc.
    """
    BASE_URL = "https://api.scryfall.com"
    RATE_LIMIT_DELAY = 0.1
    
    try:
        # Clean up card name
        clean_name = card_name.split('//')[0].strip()
        # Remove quantity prefix if present
        clean_name = ' '.join(clean_name.split()[1:] if clean_name.split()[0].isdigit() else clean_name.split())
        
        response = requests.get(
            f"{BASE_URL}/cards/named",
            params={"fuzzy": clean_name},
            timeout=10
        )
        
        if response.status_code == 200:
            time.sleep(RATE_LIMIT_DELAY)
            return response.json()
        else:
            return {"error": f"Failed to fetch data for card: {card_name}"}
            
    except Exception as e:
        return {"error": f"Error fetching data for card {card_name}: {str(e)}"}

def process_card(state: State) -> Dict:
    """
    Process the next card in the list and update state.
    """
    print(f"Processing card at index {state['current_card_index']}")
    print(f"Total cards in list: {len(state['cards'])}")
    
    if state["current_card_index"] >= len(state["cards"]):
        print("All cards have been processed.")
        return {"messages": ["All cards have been processed."]}
    
    current_card = state["cards"][state["current_card_index"]]
    print(f"Fetching data for card: {current_card}")
    card_data = fetch_scryfall_data(current_card)
    
    if "error" not in card_data:
        message = f"Processed {current_card}: {card_data.get('name')} - {card_data.get('mana_cost', 'N/A')}"
        state["card_data"].append(card_data)
        print(f"Successfully processed card: {message}")
    else:
        message = f"Failed to process {current_card}: {card_data['error']}"
        print(f"Error processing card: {message}")
    
    return {
        "messages": [message],
        "current_card_index": state["current_card_index"] + 1,
        "card_data": state["card_data"]
    }

def should_continue(state: State) -> str:
    """
    Determine if we should continue processing cards or end.
    """
    result = "process_card" if state["current_card_index"] < len(state["cards"]) else "end"
    print(f"Should continue? Current index: {state['current_card_index']}, Total cards: {len(state['cards'])}, Decision: {result}")
    return result

def create_chat_interface():
    """
    Create and configure the chat interface with file upload
    """
    with gr.Blocks() as demo:
        gr.Markdown("# AI Assistant with File Upload")
        
        with gr.Row():
            upload_button = gr.UploadButton(
                "Upload Text File",
                file_types=["text"],
                file_count="single"
            )
        
        chatbot = gr.ChatInterface(
            fn=chat_response,
            title="",
            description="An AI assistant powered by LangChain and OpenAI",
            examples=[
                "What is machine learning?",
                "Can you explain how neural networks work?",
                "What are the best practices for writing clean code?"
            ],
            theme="soft"
        )
        
        # Handle file upload
        upload_button.upload(
            fn=process_uploaded_file,
            inputs=[upload_button],
            outputs=[chatbot.textbox]  # Send file content to chat input
        )
    
    return demo

if __name__ == "__main__":
    demo = create_chat_interface()
    demo.launch()
