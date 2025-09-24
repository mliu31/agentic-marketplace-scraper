import asyncio
import sys
from typing import Optional, List, Dict, Any
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

class MCPHost:
    """MCP Host managing multiple client sessions."""
    
    def __init__(self):
        self.sessions: Dict[str, ClientSession] = {}
        self.tool_to_session_map: Dict[str, ClientSession] = {}
        self.tools: List[Dict[str, Any]] = []
        self.exit_stack = AsyncExitStack()
        self.anthropic = Anthropic()
        self.conversation_history = []

    async def connect_to_server(self, server_script_path: str):
        """Connect to a server and add its tools to the host."""
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        
        if not (is_python or is_js):
            raise ValueError("server script must be .py or .js")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(command=command, args=[server_script_path], env=None)

        # Each server gets its own transport and session
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        stdio, write = stdio_transport
        session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))
        
        await session.initialize()
        
        # Store session
        self.sessions[server_script_path] = session
        
        # Get tools for this server
        response = await session.list_tools()
        server_tools = [{
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.inputSchema
        } for tool in response.tools]
        
        print(f"Connected to {server_script_path} with tools:")
        for tool in server_tools:
            print(f"---name: {tool['name']}")
            self.tools.append(tool)
            self.tool_to_session_map[tool['name']] = session
            
    async def process_query(self, query: str) -> str:
        self.conversation_history.append({"role": "user", "content": query})
        log_parts = []

        while True:
            claude_response = self.anthropic.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                messages=self.conversation_history,
                tools=self.tools
            )

            assistant_content = []
            tool_calls_to_execute = []
            for content in claude_response.content:
                if content.type == 'text':
                    assistant_content.append(content)
                    log_parts.append(content.text)
                elif content.type == 'tool_use':
                    assistant_content.append(content)
                    tool_calls_to_execute.append(content)
            
            self.conversation_history.append({"role": "assistant", "content": assistant_content})

            if not tool_calls_to_execute:
                break

            tool_results = []
            for tool_call in tool_calls_to_execute:
                log_parts.append(f"CALLING TOOL: {tool_call.name} WITH {tool_call.input}")
                
                target_session = self.tool_to_session_map.get(tool_call.name)

                if not target_session:
                    result_text = f"Error: Tool '{tool_call.name}' not found."
                else:
                    result = await target_session.call_tool(tool_call.name, tool_call.input)
                    result_content = []
                    if result.content:
                        for content in result.content:
                            result_content.append(getattr(content, 'text', str(content)))
                    result_text = '\n'.join(result_content) if result_content else "no result"
                
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_call.id,
                    "content": result_text
                })

            self.conversation_history.append({"role": "user", "content": tool_results})
        
        return '\n'.join(log_parts)

    async def chat_loop(self):
        print("\nMCP Host started with all connected servers.")
        print("Type queries or 'quit' to exit.")
        
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
        print("usage: python client.py <path_to_server1> <path_to_server2> ...")
        sys.exit(1)

    client_host = MCPHost()
    try:
        for server_path in sys.argv[1:]:
            await client_host.connect_to_server(server_path)
        
        await client_host.chat_loop()
    finally:
        await client_host.cleanup()

if __name__ == "__main__":
    asyncio.run(main())