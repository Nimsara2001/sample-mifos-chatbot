import asyncio
import shutil
import nest_asyncio
from agents import gen_trace_id, trace
from dotenv import load_dotenv

from agent_controller import AgentController

load_dotenv()


async def main():
    async with AgentController() as controller:
        trace_id = gen_trace_id()
        # with trace(workflow_name="Mifos Confluence Workflow", trace_id=trace_id):
        #     print(f"View trace: https://platform.openai.com/traces/trace?trace_id={trace_id}\n")
        #
        #     # Example: Search and retrieve all pages in RES space
        #     result = await controller.run_confluence_workflow(
        #         "list all pages related to Mifos X"
        #     )
        #     print(result)

        with trace(workflow_name="Mifos GitHub Workflow", trace_id=trace_id):
            print(f"View trace: https://platform.openai.com/traces/trace?trace_id={trace_id}\n")

            # Example: Search and retrieve all pages in RES space
            result = await controller.run_github_workflow(
                "all markdown files related to mifos web",
            )
            print(result)


if __name__ == "__main__":
    # Check for npx installation
    if not shutil.which("npx"):
        raise RuntimeError("npx is not installed. Please install it with `npm install -g npx`.")

    # Apply nest_asyncio to fix event loop issues
    nest_asyncio.apply()

    asyncio.run(main())
