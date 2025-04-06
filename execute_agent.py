import asyncio
import shutil
import nest_asyncio
from agents import gen_trace_id, trace
from dotenv import load_dotenv
from typing import List

from agent_controller import AgentController

load_dotenv()


async def execute_confluence_agent(queries: List[str]) -> List[str]:
    """
    Execute queries on the Confluence agent and return the results.

    Args:
        queries: A list of query strings to run against Confluence

    Returns:
        A list of results corresponding to each query
    """
    results = []
    async with AgentController() as controller:
        for query in queries:
            trace_id = gen_trace_id()
            with trace(workflow_name="Mifos Confluence Workflow", trace_id=trace_id):
                print(f"View trace: https://platform.openai.com/traces/trace?trace_id={trace_id}\n")
                print(f"Executing Confluence query: {query}")

                result = await controller.run_confluence_workflow(query)
                results.append(result)
                print(f"Query result: {result}\n")

    return results


async def execute_github_agent(queries: List[str]) -> List[str]:
    """
    Execute queries on the GitHub agent and return the results.

    Args:
        queries: A list of query strings to run against GitHub

    Returns:
        A list of results corresponding to each query
    """
    results = []
    async with AgentController() as controller:
        for query in queries:
            trace_id = gen_trace_id()
            with trace(workflow_name="Mifos GitHub Workflow", trace_id=trace_id):
                print(f"View trace: https://platform.openai.com/traces/trace?trace_id={trace_id}\n")
                print(f"Executing GitHub query: {query}")

                result = await controller.run_github_workflow(query)
                results.append(result)
                print(f"Query result: {result}\n")

    return results


if __name__ == "__main__":
    # Check for npx installation
    if not shutil.which("npx"):
        raise RuntimeError("npx is not installed. Please install it with `npm install -g npx`.")

    # Apply nest_asyncio to fix event loop issues
    nest_asyncio.apply()

    # Example usage:
    confluence_queries = ["list all pages related to google summer of code",
                          "find documentation about fineract api"]
    github_queries = ["all markdown files related to mifos web",
                      "find installation instructions for community-app"]

    # Run queries (uncomment to execute)
    asyncio.run(execute_confluence_agent(confluence_queries))
    asyncio.run(execute_github_agent(github_queries))
