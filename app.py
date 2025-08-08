# app.py

from flask import Flask, request, Response, stream_with_context, send_from_directory
from flask_cors import CORS
import asyncio
import json
from langchain_core.messages import HumanMessage

# Import the graph creation function from your chatbot script
from chatbot import get_graph

# The 'static_folder' argument tells Flask where to find your frontend files.
app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

# Create the chatbot graph instance when the app starts
chatbot_graph = get_graph()

def sync_stream_wrapper(user_input, thread_id):
    """
    A synchronous generator that runs the async generator in a separate event loop.
    """
    async_gen = get_response_stream(user_input, thread_id)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        while True:
            chunk = loop.run_until_complete(async_gen.__anext__())
            yield chunk
    except StopAsyncIteration:
        pass
    finally:
        loop.close()

async def get_response_stream(user_input, thread_id):
    """
    Invokes the graph and yields chunks of the response for streaming.
    """
    config = {"configurable": {"thread_id": thread_id}}
    human_message = HumanMessage(content=user_input)
    async for event in chatbot_graph.astream_events(
        {"messages": [human_message]},
        config=config,
        version="v1"
    ):
        kind = event["event"]
        if kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                yield f"data: {json.dumps({'content': content})}\n\n"
        elif kind == "on_tool_start":
            tool_name = event['name']
            yield f"data: {json.dumps({'tool_start': tool_name})}\n\n"
        elif kind == "on_tool_end":
            tool_name = event['name']
            yield f"data: {json.dumps({'tool_end': tool_name})}\n\n"


@app.route('/')
def serve_index():
    """Serve the main HTML file from the 'static' folder."""
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """
    The main chat endpoint. It receives a message and streams back the response.
    """
    data = request.get_json()
    user_input = data.get('message')
    thread_id = data.get('thread_id', 'default_thread')
    if not user_input:
        return "Invalid request", 400
    stream = sync_stream_wrapper(user_input, thread_id)
    return Response(stream_with_context(stream), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
