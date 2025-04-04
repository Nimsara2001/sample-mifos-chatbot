import asyncio
import os
import shutil
from typing import Dict, Any, List

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from dotenv import load_dotenv

load_dotenv()

async def connect_to_server(server_name: str, command: str, args: List[str], env: Dict[str, str] = None) -> ClientSession:
    """Connect to an MCP server and return the client session."""
    print(f"Connecting to {server_name}...")
    
    server_params = StdioServerParameters(
        command=command,
        args=args,
        env=env or {}
    )
    
    # Use async with instead of await for the context manager
    async with stdio_client(server_params) as (read, write):
        session = ClientSession(read, write)
        await session.initialize()
        
        # List available tools
        tools_response = await session.list_tools()
        print(f"Available tools in {server_name}:")
        for tool in tools_response.tools:
            print(f"  - {tool.name}: {tool.description}")
        
        return session

async def call_tool(session: ClientSession, tool_name: str, tool_args: Dict[str, Any]) -> Any:
    """Call a specific tool on the MCP server."""
    print(f"Calling tool '{tool_name}' with args: {tool_args}")
    result = await session.call_tool(tool_name, tool_args)
    return result.content

async def main():
    # Check if npx is installed
    if not shutil.which("npx"):
        raise RuntimeError("npx is not installed. Please install it with `npm install -g npx`.")
    
    # Check if uvx is installed
    if not shutil.which("uvx"):
        raise RuntimeError("uvx is not installed. Please install it.")
    
    # Connect to GitHub server
    github_session = await connect_to_server(
        server_name="GitHub Server",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-github"],
        env={"GITHUB_PERSONAL_ACCESS_TOKEN": os.getenv('GITHUB_PERSONAL_ACCESS_TOKEN')}
    )
    
    # Connect to Atlassian server
    atlassian_session = await connect_to_server(
        server_name="Atlassian Server",
        command="uvx",
        args=[
            "mcp-atlassian",
            f"--confluence-url=https://mifosforge.jira.com/wiki",
            f"--confluence-username=mihin.nimsara.2001@gmail.com",
            f"--confluence-token={os.getenv('CONFLUENCE_API_TOKEN')}"
        ]
    )
    
    try:
        # Example: Call GitHub tool to search repositories
        github_result = await call_tool(
            github_session, 
            "search_repositories", 
            {"query": "GSoC 2025", "per_page": 5}
        )
        print("\nGitHub Search Results:")
        print(github_result.text)
        
        # Example: Call Confluence tool to get page content
        confluence_result = await call_tool(
            atlassian_session,
            "get_page_by_title",
            {"space_key": "GSOC", "title": "Ideas"}
        )
        print("\nConfluence Page Content:")
        print(confluence_result.text)
        
    finally:
        # Clean up sessions
        await github_session.close()
        await atlassian_session.close()

if __name__ == "__main__":
    asyncio.run(main())
