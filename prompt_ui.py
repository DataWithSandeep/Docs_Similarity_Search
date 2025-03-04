from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate, load_prompt
import time

load_dotenv()
# model=ChatGroq(model="gemma2-9b-it")
model=ChatOpenAI(model='gpt-4')

st.header('Research Tool')
# user_input=st.text_input('Enter your prompt')
paper_input = st.selectbox( "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )

#template
template=load_prompt('myprompt.json')

# prompt=template.invoke({
#         'paper_input':paper_input,
#         'style_input':style_input,
#         'length_input':length_input})

## now here we use two invoke system which is not a ideal for way in langchain ecosystem.
## so we use now langchain chai feature to overcome this.

if st.button('Response'):
    chain=template | model
    result=chain.invoke({
        'paper_input':paper_input,
        'style_input':style_input,
        'length_input':length_input})
    # result=model.invoke(prompt)
    st.write(result.content)

  
## if you want to strem your response.
# if st.button('Response'):
#     response_container = st.empty()
#     chain = template | model
#     full_response = ""
#     for chunk in chain.stream(
#         {
#             "paper_input": paper_input,
#             "style_input": style_input,
#             "length_input": length_input
#         }
#     ):
#         full_response += chunk.content + " "  # Append content with a space
#         time.sleep(0.01)
#         response_container.write(full_response)