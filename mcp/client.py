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
    """
    This class now acts as an MCP Host, managing multiple client sessions.
    """
    def __init__(self):
        # NEW: Store multiple sessions, keyed by server path for identification.
        self.sessions: Dict[str, ClientSession] = {} 
        # NEW: A map to find which session a tool belongs to.
        self.tool_to_session_map: Dict[str, ClientSession] = {}
        # NEW: A single, aggregated list of all tools from all servers.
        self.tools: List[Dict[str, Any]] = []

        self.exit_stack = AsyncExitStack()
        self.anthropic = Anthropic()
        self.conversation_history = []

    async def connect_to_server(self, server_script_path: str):
        """
        Connects to a single server and adds its session and tools to the host's state.
        """
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
        
        # NEW: Store the session in our dictionary
        self.sessions[server_script_path] = session
        
        # Get tools for THIS server
        response = await session.list_tools()
        server_tools = [{
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.inputSchema
        } for tool in response.tools]
        
        print(f"Connected to {server_script_path} with tools:")
        # NEW: Add this server's tools to our aggregated list and map
        for tool in server_tools:
            print(f"---name: {tool['name']}")
            self.tools.append(tool) # Add to the main list
            self.tool_to_session_map[tool['name']] = session # Map tool name to its session
            
    async def process_query(self, query: str) -> str:
        self.conversation_history.append({"role": "user", "content": query})
        log_parts = []

        while True:
            # The Claude API call now receives the aggregated list of all tools
            claude_response = self.anthropic.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                messages=self.conversation_history,
                tools=self.tools # Pass the combined tool list
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
                
                # NEW: Look up the correct session for this specific tool
                target_session = self.tool_to_session_map.get(tool_call.name)

                if not target_session:
                    # Handle the case where the tool is not found
                    result_text = f"Error: Tool '{tool_call.name}' not found."
                else:
                    # Call the tool using its specific session
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
        print("You can now use tools from any server.")
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

    client_host = MCPClient()
    try:
        for server_path in sys.argv[1:]:
            await client_host.connect_to_server(server_path)
        
        await client_host.chat_loop()
    finally:
        await client_host.cleanup()

if __name__ == "__main__":
    asyncio.run(main())