try:
    import os
    import asyncio
    import torch
    from openai import OpenAI
    from fastmcp import Client
    from base64 import b64encode
except ImportError:
    raise ImportError('Error importing modules. Ensure all packages from ../requirements.txt are installed. Run `pip '
          'install -r requirements.txt` in the terminal to install the packages.')

# Create ../input-files if it does not exist
input_relative_path = '../input-files'
input_abspath = os.path.normpath(
    os.path.join(os.path.dirname(__file__), input_relative_path)
)
if not os.path.exists(input_abspath):
    print(f'Input files directory at {input_abspath} does not exist, creating new directory.')
    os.mkdir(input_abspath)


# Model ID
MODEL_ID = "google/gemma-3n-e4b-it" #"alfredcs/gemma-3N-finetune"

# Create vLLM client
OPENAI_API_KEY = "EMPTY"
OPENAI_API_BASE = "http://video.cavatar.info:8087"

client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_API_BASE,
)

# Create FastMCP client
MCP_URL = 'http://localhost:3000/mcp'
mcp_client = Client(MCP_URL)


# Get tools
async def get_tools():
    try:
        async with mcp_client:
            return await mcp_client.list_tools()
    except RuntimeError:
        raise RuntimeError('Ensure master_mcp_server.py is running before starting the chatbot.')


# Call tool
async def call_tool(tool):
    async with mcp_client:
        return await mcp_client.call_tool(tool.function.name, tool.function.arguments)


# See if file is an image or audio based on the extension
def get_file_info(file_abspath: str):
    # Get file as bytes
    with open(file_abspath, "rb") as file:
        encoded_bytes = b64encode(file.read())
        encoded_string = encoded_bytes.decode('utf-8')

    # Get file extension
    _, extension = os.path.splitext(file_abspath)

    match extension:
        case ".jpg" | ".png" | ".webp":
            return "image", encoded_string
        case ".mp3" | ".wav":
            return "audio", encoded_string
        case _:
            return "UNKNOWN", encoded_string


# Process a query
async def process_query(query: str):
    global messages

    # Add relevant files to message content
    message_content = []
    for entry in os.listdir(input_abspath):
        entry_abspath = os.path.join(input_abspath, entry)

        # Continue if entry is a directory
        if not os.path.isfile(entry_abspath):
            continue

        # Continue if file is marked with a #
        if entry.startswith("#"):
            continue

        # Add message content based on file type
        file_type, file_b64 = get_file_info(entry_abspath)

        message_content.append({"type": file_type, file_type: file_b64})

    # Add query to message content
    message_content.append({"type": "text", "text": query})

    # Add all content to message memory
    messages.append({"role": "user", "content": message_content})

    # Call model with messages
    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=messages,
        tools=tools
    )
    tool_calls = response.choices[0].message.tool_calls
    print(f'Selected tools: {tool_calls}')

    # Call tools and save results
    if len(tool_calls):
        for tool in tool_calls:
            result = await call_tool(tool)
            messages.append({"role": "user", "content": result.content[0].text})

    # Get next response from LLM
    response = client.chat.completions.create(
        model=MODEL_ID,
        max_tokens=3000,
        messages=messages,
    )
    response = response.choices[0].message.content
    print("Chat response:", response)

    # Save model output to message memory
    messages.append({"role": "assistant", "content": response})
    return response


# Main chat loop
messages = []
tools = []
async def main():
    global messages, tools

    # Get tools
    tools = await get_tools()

    while True:
        # try:
        query = input("\nQuery: ").strip()

        # Exit the program if the user enters "quit"
        if query.lower() == "quit":
            break

        # Wipe all message memory if the user enters "wipe"
        if query.lower() == "wipe":
            messages = []
            continue

        response = await process_query(query)
        print("\n" + response)

        # except Exception as e:
        #     print(f"\n{type(e).__name__}: {str(e)}")


if __name__ == '__main__':
    asyncio.run(main())
