import asyncio
import os
import shutil
import sys
import nest_asyncio

from agents import Agent, Runner, gen_trace_id, trace
from agents.mcp import MCPServer, MCPServerStdio

from dotenv import load_dotenv
load_dotenv()



# ...existing code...

async def run(mcp_server: MCPServer):
    try:
        agent = Agent(
            name="GitHub Agent",
            instructions="""Use the tools to access github repositories and do your best to answer the questions.""",
            mcp_servers=[mcp_server],
        )
        
        print("Starting agent process..............................................................")

        # List the files it can read
        # message = "https://github.com/modelcontextprotocol/servers.git list latest 5 issues in this repo"
        # print(f"Running: {message}")
        # result = await Runner.run(starting_agent=agent, input=message)
        # print(result.final_output)
        
        message = "https://github.com/openMF/web-app.git can you explain this proejct?"
        print(f"Running: {message}")
        result = await Runner.run(starting_agent=agent, input=message)
        print(result.final_output)
        
        print("Agent process finished..............................................................")
        
    except Exception as e:
        print(f"Error during execution: {e}")
        raise


async def main():
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    # samples_dir = os.path.join(current_dir, "sample_files")

    async with MCPServerStdio(
        name="github server via npx",
        params={
            "command": "npx",
            "args": [
                "-y",
                "@modelcontextprotocol/server-github"
            ],
            "env": {
                "GITHUB_PERSONAL_ACCESS_TOKEN": "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"
            }
        },
    ) as server:
        trace_id = gen_trace_id()
        with trace(workflow_name="MCP Github Test", trace_id=trace_id):
            print(f"View trace: https://platform.openai.com/traces/trace?trace_id={trace_id}\n")
            await run(server)


if __name__ == "__main__":
    # Let's make sure the user has npx installed
    if not shutil.which("npx"):
        raise RuntimeError("npx is not installed. Please install it with `npm install -g npx`.")
    
    # Apply nest_asyncio to fix event loop issues
    nest_asyncio.apply()
    
    asyncio.run(main())
    