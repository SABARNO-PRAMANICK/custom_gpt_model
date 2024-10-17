import torch
import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel 

model_name = "gpt2"  
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def generate_text(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    output = model.generate(
        input_ids,
        max_length=80,               # Shorten length for more focused responses
        num_beams=7,                 # Use beam search with 7 beams
        no_repeat_ngram_size=2,      # Avoid repetition
        temperature=0.7,             # Reduce randomness
        top_k=50,                    # Sample from top 50 tokens
        top_p=0.9,                   # Use nucleus sampling
        repetition_penalty=1.2,      # Penalize repeated phrases
        early_stopping=True
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Title and introduction
st.title("Custom GPT Language Model")
st.markdown("Welcome to the GPT-based language model platform. Start a conversation below:")

# Initialize session state for storing the conversation history
if "generated_responses" not in st.session_state:
    st.session_state["generated_responses"] = []
if "past_prompts" not in st.session_state:
    st.session_state["past_prompts"] = []

# User input section
user_input = st.text_input("You: ", key="user_input")

# When the user inputs a prompt
if user_input:
    prompt = f"Nova is a helpful, friendly assistant. Nova says: {user_input}"
    
    # Generate a response
    with st.spinner("Generating response..."):
        response = generate_text(prompt)

    # Store the prompt and the response
    st.session_state.past_prompts.append(user_input)
    st.session_state.generated_responses.append(response)

# Display the conversation history
if st.session_state["generated_responses"]:
    for i in range(len(st.session_state["generated_responses"])):
        st.markdown(f"**You**: {st.session_state['past_prompts'][i]}")
        st.markdown(f"**AI**: {st.session_state['generated_responses'][i]}")
        st.markdown("---")

# Function to add custom CSS styles
# Function to add custom CSS styles
def add_css():
    st.markdown("""
    <style>
    body {
        background-color: #1e1e1e; /* Dark background */
        color: #ffffff; /* Text color */
    }
    .stTextInput label {
        font-size: 16px;
        font-weight: bold;
        color: #b100cd; /* Change label color */
    }
    .stTextInput div {
        background-color: #b100cd; /* Input box background */
        border-radius: 5px;
        padding: 10px;
    }
    .stMarkdown {
        font-size: 16px;
        color: #ffffff; /* Markdown text color */
    }
    .stButton button {
        background-color: #b100cd; /* Button color */
        color: white;
        font-size: 16px;
    }
    .stButton button:hover {
        background-color: #a100bd; /* Darker shade on hover */
    }
    </style>
    """, unsafe_allow_html=True)

