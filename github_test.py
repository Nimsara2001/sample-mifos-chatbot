import asyncio
import os
import shutil
import nest_asyncio

from agents import Agent, ModelSettings, Runner, gen_trace_id, trace
from agents.mcp import MCPServer, MCPServerStdio

from dotenv import load_dotenv
load_dotenv()


async def run(github_server: MCPServer, atlassian_server: MCPServer):
    try:
        agent = Agent(
            name="GitHub and Atlassian Agent",
            instructions="""Use the tools to access github repositories and Atlassian Confluence to do your best to answer the questions.""",
            mcp_servers=[github_server, atlassian_server]
        )
        
        print("Starting agent process..............................................................")
        
        message = "list all Mifos documents in atlassian confluence and list all repositories in github"
        print(f"Running: {message}")
        result = await Runner.run(starting_agent=agent, input=message)
        print(result.final_output)
        
        print("Agent process finished..............................................................")
        
    except Exception as e:
        print(f"Error during execution: {e}")
        raise


async def main():
    async with MCPServerStdio(
        name="github server via npx",
        params={
            "command": "npx",
            "args": [
                "-y",
                "@modelcontextprotocol/server-github"
            ],
            "env": {
                "GITHUB_PERSONAL_ACCESS_TOKEN": os.getenv('GITHUB_PERSONAL_ACCESS_TOKEN')
            }
        },
    ) as github_server, MCPServerStdio(
        name="atlassian server via uvx",
        params={
            "command": "uvx",
            "args": [
                "mcp-atlassian",
                "--confluence-url=https://mifosforge.jira.com/wiki",
                "--confluence-username=mihin.nimsara.2001@gmail.com",
                "--confluence-token=" + os.getenv('CONFLUENCE_API_TOKEN')
            ]
        }
    ) as atlassian_server:
        trace_id = gen_trace_id()
        with trace(workflow_name="MCP Github and Atlassian Test", trace_id=trace_id):
            print(f"View trace: https://platform.openai.com/traces/trace?trace_id={trace_id}\n")
            await run(github_server, atlassian_server)


if __name__ == "__main__":
    # Let's make sure the user has npx installed
    if not shutil.which("npx"):
        raise RuntimeError("npx is not installed. Please install it with `npm install -g npx`.")
    
    # Apply nest_asyncio to fix event loop issues
    nest_asyncio.apply()
    
    asyncio.run(main())
