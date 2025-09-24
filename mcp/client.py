import asyncio
import sys
from typing import Optional, List, Dict, Any
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.anthropic = Anthropic()
        self.conversation_history = []

    async def connect_to_server(self, server_script_path: str):
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        
        if not (is_python or is_js):
            raise ValueError("server script must be .py or .js")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        
        await self.session.initialize()
        
        # cache tools for reuse
        response = await self.session.list_tools()
        self.tools = [{
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.inputSchema
        } for tool in response.tools]
        
        print("connected with tools:")
        for tool in self.tools:
            print(f"---name: {tool['name']} \n    description: {tool['description']} \n    input_schema: {tool['input_schema']}")

    async def process_query(self, query: str) -> str:
        """Process query with Claude using atomic message blocks - complete assistant responses followed by complete tool results"""
        # add query to history
        self.conversation_history.append({"role": "user", "content": query})
        log_parts = []

        while True:
            # claude api call with messages=history
            claude_response = self.anthropic.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                messages=self.conversation_history,
                tools=self.tools
            )

            # collect ALL assistant content first
            assistant_content = []
            tool_calls_to_execute = []
            tool_count = 0

            # for each content in response.content
            for content in claude_response.content:
                if content.type == 'text':
                    assistant_content.append(content)
                    log_parts.append(content.text)
                elif content.type == 'tool_use':
                    assistant_content.append(content)
                    tool_calls_to_execute.append(content)
                    tool_count += 1

            # add complete assistant message to history
            self.conversation_history.append({
                "role": "assistant", 
                "content": assistant_content
            })

            # execute all tools, collect all results
            tool_results = []
            for tool_call in tool_calls_to_execute:
                # log tool call
                log_parts.append(f"calling {tool_call.name} with {tool_call.input}")
                
                # call tool
                result = await self.session.call_tool(tool_call.name, tool_call.input)
                
                # format result content
                result_content = []
                if result.content:
                    for content in result.content:
                        if hasattr(content, 'text'):
                            result_content.append(content.text)
                        else:
                            result_content.append(str(content))
                
                result_text = '\n'.join(result_content) if result_content else "no result"
                
                # collect tool result
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_call.id,
                    "content": result_text
                })

            # add all tool results as one user message
            if tool_results:
                self.conversation_history.append({
                    "role": "user",
                    "content": tool_results
                })

            # if tool count == 0: break
            if tool_count == 0:
                break

        # return log joined by new lines
        return '\n'.join(log_parts)

    async def chat_loop(self):
        print("\nmcp client started")
        print("type queries or 'quit' to exit")
        
        while True:
            try:
                query = input("\nquery: ").strip()
                if query.lower() == 'quit':
                    break
                
                response = await self.process_query(query)
                print(f"\n{response}")
                
            except Exception as e:
                print(f"error: {str(e)}")

    async def cleanup(self):
        await self.exit_stack.aclose()

async def main():
    if len(sys.argv) < 2:
        print("usage: python client.py <path_to_server_script1> <path_to_server_script2> ...")
        sys.exit(1)

    client = MCPClient()
    try:
        # Connect to all provided servers
        for i, server_path in enumerate(sys.argv[1:], 1):
            await client.connect_to_server(server_path)
        
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())