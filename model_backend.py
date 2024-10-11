import torch
import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel # type: ignore

# Load the pre-trained GPT2 model and tokenizer
model_name = "gpt2"  # You can replace this with your custom model
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to generate text
def generate_text(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    output = model.generate(input_ids, max_length=150, num_beams=5,
                            no_repeat_ngram_size=2, early_stopping=True)
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
    # Generate a response
    with st.spinner("Generating response..."):
        response = generate_text(user_input)

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

