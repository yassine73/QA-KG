from langchain_core.messages import HumanMessage, AIMessage
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from inference.agent import HybridAgent

st.title('QA Knowledge Graph')

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'hybrid_agent' not in st.session_state:
    st.session_state.hybrid_agent = None

if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None

if not st.session_state.hybrid_agent:
    uploaded_file = st.sidebar.file_uploader("Choose a file", type=['TXT', 'PDF'])
    if uploaded_file:
        with st.spinner('Loading...'):
            st.session_state.hybrid_agent = HybridAgent(file=uploaded_file)  
    else:
        st.session_state.messages = []
        st.session_state.hybrid_agent = None
        st.write('Please upload a document to start')

if st.session_state.hybrid_agent:
    question = st.chat_input('Ask a question')
    if question:
        st.session_state.messages.append(HumanMessage(content=question))
        response = st.session_state.hybrid_agent.ask(st.session_state.messages)
        st.session_state.messages.append(AIMessage(content=question))
        st.write(response)
    
    