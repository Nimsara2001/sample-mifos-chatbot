import asyncio
import os
import shutil
from typing import Dict, Any, List
import traceback

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from dotenv import load_dotenv

load_dotenv()

async def main():
    # Check if required tools are installed
    if not shutil.which("npx"):
        raise RuntimeError("npx is not installed. Please install it with `npm install -g npx`.")
    
    if not shutil.which("uvx"):
        raise RuntimeError("uvx is not installed. Please install it.")
    
    print("Connecting to Atlassian Server...")
    
    server_params = StdioServerParameters(
        command="uvx",
        args=[
            "mcp-atlassian",
            f"--confluence-url=https://mifosforge.jira.com/wiki",
            f"--confluence-username=mihin.nimsara.2001@gmail.com",
            f"--confluence-token={os.getenv('CONFLUENCE_API_TOKEN')}"
        ],
        env={}
    )
    
    try:
        # Connect to the server
        async with stdio_client(server_params) as (read, write):
            session = ClientSession(read, write)
            await session.initialize()
            
            # List available tools
            tools_response = await session.list_tools()
            print("Available tools in Atlassian Server:")
            for tool in tools_response.tools:
                print(f"  - {tool.name}: {tool.description}")
            
            # Example: Call Confluence tool to get page content
            print("\nCalling tool 'confluence_get_page'...")
            result = await session.call_tool(
                "confluence_get_page",
                {
                    "page_id": "4271669249"
                }
            )
            
            # Print the result - adjust based on actual structure
            print("\nConfluence Page Content:")
            if hasattr(result.content, 'text'):
                print(result.content.text)
            else:
                print(result.content)  # Might be a dictionary or other structure
                
    except Exception as e:
        print(f"Error: {e}")
        print(traceback.format_exc())

if __name__ == "__main__":
    asyncio.run(main())
