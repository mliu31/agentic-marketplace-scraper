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

    async def connect_to_server(self, server_cmd: str):
        """Connect to a server and add its tools to the host."""
        if not server_cmd:
            raise ValueError("empty server command")
        server_params = StdioServerParameters(command=server_cmd[0], args=server_cmd[1:], env=None)

        # Each server gets its own transport and session
        transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        stdio, write = transport
        session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))
        
        try:
            await session.initialize()
        except Exception as e:
            raise RuntimeError(f"failed to initialize server: {' '.join(server_cmd)}") from e
        
        # Store session
        self.sessions[" ".join(server_cmd)] = session
        
        # Get tools for this server
        resp = await session.list_tools()
        for t in resp.tools:
            tool = {"name": t.name, "description": t.description, "input_schema": t.inputSchema}
            self.tools.append(tool)
            self.tool_to_session_map[t.name] = session
        print(f"connected: {' '.join(server_cmd)} -> {[t.name for t in resp.tools]}")
    
    def truncate_history(self, max_messages=50):
            """Keep conversation history under max_messages to prevent context overflow"""
            if len(self.conversation_history) > max_messages:
                self.conversation_history = self.conversation_history[-max_messages:]
            
    async def process_query(self, query: str) -> str:
        """Process query with Claude using atomic message blocks - complete assistant responses followed by complete tool results"""
        self.truncate_history()
        self.conversation_history.append({"role": "user", "content": query})
        log_parts = []

        while True:
            claude_response = self.anthropic.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                messages=self.conversation_history,
                tools=self.tools
            )

            # Parse model response
            assistant_content = []
            tool_calls_to_execute = []
            for content in claude_response.content:
                assistant_content.append(content)
                if content.type == 'text':
                    log_parts.append(content.text)
                elif content.type == 'tool_use':
                    tool_calls_to_execute.append(content)
            
            # Log entire LLM response as one assistant message
            self.conversation_history.append({"role": "assistant", "content": assistant_content})

            # If no tool calls, we've completed the assistant message
            if not tool_calls_to_execute:
                break

            # Execute tool calls 
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

            # Log all tool call results as one user message
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
    servers = [[sys.executable, "C:/Users/mgnli/agentic-marketplace-scraper/mcp/weather.py"], 
    [sys.executable, "C:/Users/mgnli/agentic-marketplace-scraper/mcp/weather2.py"], 
    ["uv", "--directory","C:/Users/mgnli/gmail-mcp-server", "run", "gmail", "--creds-file-path", "C:/Users/mgnli/gmail-mcp-server/gmail_creds.json", "--token-path", "C:/Users/mgnli/gmail-mcp-server/gmail_tokens.json"]]

    client_host = MCPHost()
    try:
        for server_params in servers:
            await client_host.connect_to_server(server_params)
        
        await client_host.chat_loop()
    finally:
        await client_host.cleanup()

if __name__ == "__main__":
    asyncio.run(main())