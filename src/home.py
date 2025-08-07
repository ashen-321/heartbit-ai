import streamlit as st
import io
from time import time
import asyncio
from util import *
from openai import OpenAI
from fastmcp import Client
from base64 import b64encode


# --------------------------------------------------------------------------------------------
# Webpage Setup ------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------


# Variables and constants
aoss_host = read_key_value(".aoss_config.txt", "AOSS_host_name")
aoss_index = read_key_value(".aoss_config.txt", "AOSS_index_name")

input_file_path = os.path.join(os.path.dirname(__file__), "input-files")
image_extensions = [".jpg", ".jpeg", ".png", ".webp"]
input_image_file = "input_image"
query_audio_file = "query_audio.wav"
last_uploaded_files = None

os.environ["OPENAI_API_KEY"] = "EMPTY"
os.environ["OPENAI_BASE_URL"] = "http://video.cavatar.info:8087/v1"
MODEL_ID = 'alfredcs/gemma-3N-finetune'
voice_prompt = ''
SYSTEM_PROMPT = 'You are a helpful assistant. Please answer the user question accurately and truthfully.'
DISPLAYED_PROMPT = 'I am your assistant. How can I help today?'

MCP_URL = 'http://localhost:3000/mcp'

# Streamlit setup
st.set_page_config(page_title="Gemma-3N", page_icon="🩺", layout="wide")
st.title("Personal assistant")


# --------------------------------------------------------------------------------------------
# Sidebar ------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------


with st.sidebar:
    st.header(':green[Settings]')

    # File upload box
    upload_file = st.file_uploader("Upload your images here:", accept_multiple_files=True, type=["jpg", "jpeg", "png", "webp"])

    # Only update input file directory if something has changed
    if upload_file != last_uploaded_files:
        # File saving
        image_file_indexes = []

        # Clear input file directory
        empty_directory(input_file_path)

        # Save file type indexes
        if upload_file is not None:
            for i in range(len(upload_file)):
                _, upload_file_extension = os.path.splitext(upload_file[i].name)
                if upload_file_extension in image_extensions:
                    image_file_indexes.append(i)

        # Read file indexes and save accordingly
        # Image upload
        for i in range(len(image_file_indexes)):
            index = image_file_indexes[i]
            image_bytes = upload_file[index].read()
            st.image(io.BytesIO(image_bytes))

            input_file = os.path.join(input_file_path, input_image_file + f"_{i}" + upload_file_extension)
            with open(input_file, 'wb') as image_file:
                image_file.write(image_bytes)

        last_uploaded_files = upload_file

        # Configuration sliders
        max_tokens = st.number_input("Maximum Output Tokens", min_value=0, value=4096, max_value=4096, step=64)

        # --- Audio query -----#
        st.divider()
        st.header(':green[Enable voice input]')
        record_audio_bytes = st.audio_input("Toggle mic to start/stop recording")
        if record_audio_bytes:
            with open(query_audio_file, 'wb') as audio_file:
                audio_file.write(record_audio_bytes.getvalue())
            if os.path.exists(query_audio_file):
                voice_prompt = get_asr(query_audio_file).encode('utf-8').decode('unicode_escape')

        # ---- Clear chat history ----
        st.divider()
        if st.button("Clear Chat History"):
            st.session_state.messages.clear()
            st.session_state["messages"] = [{"role": "system", "content": SYSTEM_PROMPT}]
            st.session_state["displayed_messages"] = [{"role": "assistant", "content": DISPLAYED_PROMPT}]
            record_audio_bytes = None
            voice_prompt = ""


# --------------------------------------------------------------------------------------------
# MCP ----------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------


# Create FastMCP client
if "mcp_client" not in st.session_state:
    st.session_state["mcp_client"] = Client(MCP_URL)


# Get tools
async def get_tools():
    try:
        async with st.session_state["mcp_client"] as mcp_client:
            tools_list = await mcp_client.list_tools()

            return list(map(tool_reformat, tools_list))
    except RuntimeError:
        raise RuntimeError('Ensure master_mcp_server.py is running before starting the chatbot.')


# Reformat FastMCP tools to OpenAI standard
def tool_reformat(tool):
    return {
        'type': 'function',
        'function': {
            'name': tool.name,
            'description': tool.description,
            'parameters': tool.inputSchema,
        },
        'strict': True
    }


# Call tool
async def call_tool(tool):
    async with st.session_state["mcp_client"] as mcp_client:
        return await mcp_client.call_tool(tool.function.name, tool.function.arguments)


# --------------------------------------------------------------------------------------------
# GUI ----------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------


start_time = time()

# Message tracking
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "system", "content": SYSTEM_PROMPT}]
    st.session_state["displayed_messages"] = [
        {"role": "assistant", "content": "I am your assistant. How can I help today?"}]

for msg in st.session_state.displayed_messages:
    st.chat_message(msg["role"]).write(msg["content"])

# OpenAI Client
if "openai_client" not in st.session_state:
    st.session_state["openai_client"] = OpenAI()

# Prompt input logic
if prompt := st.chat_input() or len(voice_prompt) > 3:
    prompt_flag = isinstance(prompt, str)

    # Override query with voice prompt if it is missing
    if not prompt_flag:
        prompt = voice_prompt

    # Get tools
    tools = asyncio.run(get_tools())

    # Add relevant files to message content
    message_content = []
    with_files = False
    for entry in os.listdir(input_file_path):
        entry_abspath = os.path.join(input_file_path, entry)
        file_type = None
        file_contents = None

        # Get file as bytes
        with open(entry_abspath, "rb") as file:
            file_bytes = b64encode(file.read()).decode('utf-8')

        # Format bytes for OpenAI standard
        if "image" in entry:
            file_type = "image_url"
            file_contents = {"url": f"data:image/jpeg;base64,{file_bytes}"}

        # Abort for invalid files
        if file_type is None or file_contents is None:
            continue

        # Add message content based on file type
        message_content.append({"type": file_type, file_type: file_contents})
        with_files = True

    # Add prompt to message content
    message_content.append({"type": "text", "text": prompt})

    # Add message content to messages
    st.session_state.messages.append({"role": "user", "content": message_content})
    st.session_state.displayed_messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Generate response
    response = openai_generate(MODEL_ID, st.session_state.messages, max_tokens, tools)

    # Call tools and save results
    tool_calls = response.choices[0].message.tool_calls
    if len(tool_calls):
        for tool in tool_calls:
            result = asyncio.run(call_tool(tool))
            st.session_state.messages.append({"role": "user", "content": result.content})

    # Get next response from LLM
    response = openai_generate(MODEL_ID, st.session_state.messages, max_tokens)
    footer = (f'✒︎***Content created with:*** {MODEL_ID}, Latency: {(time() - start_time) * 1000:.2f} ms, '
              f'Completion Tokens: {response.usage.completion_tokens}, Prompt Tokens: {response.usage.prompt_tokens}, '
              f'Total Tokens:{response.usage.total_tokens}')

    # Display text and save to message memory
    response = response.choices[0].message.content
    response_formatted = f"{response}\n\n {footer}"
    st.session_state.messages.append({"role": "assistant", "content": response_formatted})
    st.session_state.displayed_messages.append({"role": "assistant", "content": response_formatted})
    st.chat_message("ai", avatar='🤵').write(response_formatted)
