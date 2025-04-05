import asyncio
import os
import shutil
from mcp import ClientSession, StdioServerParameters
import nest_asyncio

from agents import Agent, Runner, gen_trace_id, trace
from agents.mcp import MCPServer, MCPServerStdio
from  mcp.client.stdio import stdio_client
from pydantic import BaseModel

from dotenv import load_dotenv
load_dotenv()


# Define structured output model
class ConfluencePage(BaseModel):
    page_id: str
    url: str
    title: str

class ConfluencePageList(BaseModel):
    results: list[ConfluencePage]

async def run(atlassian_server: MCPServer):
    try:
        # Configure agent with structured output
        agent = Agent(
            name="Atlassian Search Agent",
            instructions="""**Operation Protocol**
                1. EXCLUSIVELY execute `confluence_search()` for query resolution with adding maximum limit
                2. Directly return raw search results in specified format
                3. NEVER invoke `confluence_get_page()`

                **Output Requirements**
                - Strict JSON array from search API
                - Mandatory fields per entry:
                - page_id: string (exact match)
                - url: full Confluence URL
                - title: unmodified page title
                - No content analysis or post-processing

                **Example Output Structure:**
                [
                {
                    "page_id": "12345",
                    "url": "https://confluence.example.com/page/12345",
                    "title": "Project Documentation"
                }
                ]""",
            mcp_servers=[atlassian_server],
            # model='gpt-4.0-turbo',
            output_type=ConfluencePageList,
        )
        
        
        print("Starting agent process..............................................................")
        
        message = "get all pages in the confluence space RES. Do not miss any pages or limit pages count."
        print(f"Running: {message}")
        result = await Runner.run(starting_agent=agent, input=message)
        # Convert the ConfluencePageList to string representation
        print(str(result.final_output) + "\n--------------------------")

        try:
            # Access the results directly from the ConfluencePageList object
            pages_to_process = [
                {"page_id": page.page_id} 
                for page in result.final_output.results
            ]
            
            if pages_to_process:
                response = await run_confluence_get_page(pages_to_process)
                print(response)
            else:
                print("No valid page IDs found in the search results")
                
        except Exception as e:
            print(f"Error processing pages: {str(e)}")
                
        except Exception as e:
            print(f"Error processing pages: {str(e)}")
        
        print("Agent process finished..............................................................")
        
        
        
    except Exception as e:
        print(f"Error during execution: {e}")
        raise

async def run_confluence_get_page(pages_list):
    if not pages_list or not isinstance(pages_list, list):
        raise ValueError("pages_list must be a non-empty list")

    # Create mifos_mds directory if it doesn't exist
    output_dir = "mifos_mds"
    os.makedirs(output_dir, exist_ok=True)

    params = StdioServerParameters(
        command="uvx",
        args=[
                "mcp-atlassian",
                "--confluence-url=https://mifosforge.jira.com/wiki",
                "--confluence-username=mihin.nimsara.2001@gmail.com",
                "--confluence-token=" + os.getenv('CONFLUENCE_API_TOKEN')
            ]
    )
    try:
        async with stdio_client(params) as streams:
            async with ClientSession(streams[0], streams[1]) as session:
                await session.initialize()
                
                for page in pages_list:
                    try:
                        page_id = page.get("page_id")
                        if not page_id:
                            print(f"Skipping invalid page entry: {page}")
                            continue
                            
                        result = await session.call_tool("confluence_get_page", {"page_id": page_id})
                        text_dict = eval(result.content[0].text)
                        content = text_dict["metadata"]["content"]["value"]
                        
                        # Write content to MD file in mifos_mds folder
                        filename = os.path.join(output_dir, f'{page_id}.md')
                        with open(filename, 'w', encoding='utf-8') as md_file:
                            md_file.write(content)
                            print(f"Created file: {filename}")
                    except Exception as e:
                        print(f"Error processing page {page_id}: {str(e)}")
                
                return f"Files created successfully in {output_dir} folder"
    except Exception as e:
        raise RuntimeError(f"Failed to process confluence pages: {str(e)}")
        

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
        cache_tools_list=True,
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
        with trace(workflow_name="Atlassian Agent workflow", trace_id=trace_id):
            print(f"View trace: https://platform.openai.com/traces/trace?trace_id={trace_id}\n")
            await run(atlassian_server)


if __name__ == "__main__":
    # Let's make sure the user has npx installed
    if not shutil.which("npx"):
        raise RuntimeError("npx is not installed. Please install it with `npm install -g npx`.")
    
    # Apply nest_asyncio to fix event loop issues
    nest_asyncio.apply()
    
    asyncio.run(main())
