import os
import asyncio
from typing import List, Dict
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from agents import Runner
from agents.mcp import MCPServerStdio

from mifos_agents import MifosAgents


class AgentController:
    def __init__(self):
        self.agents = None
        self._confluence_server = None
        self._github_server = None
        self._current_task = None

    @property
    def confluence_server(self):
        return self._confluence_server

    @property
    def github_server(self):
        return self._github_server

    async def __aenter__(self):
        self._current_task = asyncio.current_task()
        await self.initialize_servers()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._current_task == asyncio.current_task():
            await self.cleanup()
        self._current_task = None

    async def initialize_servers(self):
        try:
            # Initialize Confluence server
            self._confluence_server = MCPServerStdio(
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
            )

            # Initialize GitHub server
            self._github_server = MCPServerStdio(
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
                }
            )

            # Enter server contexts in the correct task scope
            await self._confluence_server.__aenter__()
            await self._github_server.__aenter__()

            self.agents = MifosAgents(self._confluence_server, self._github_server)

        except Exception as e:
            # Ensure cleanup happens in the same task scope
            if self._current_task == asyncio.current_task():
                await self.cleanup()
            raise RuntimeError(f"Failed to initialize servers: {str(e)}")

    async def cleanup(self):
        try:
            if self._github_server:
                await self._github_server.__aexit__(None, None, None)
                self._github_server = None
            if self._confluence_server:
                await self._confluence_server.__aexit__(None, None, None)
                self._confluence_server = None
            self.agents = None
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")

    async def search_confluence(self, query: str) -> List[Dict]:
        if not self.agents:
            raise RuntimeError("Servers not initialized. Call initialize_servers() first")

        agent = self.agents.get_confluence_search_agent()
        result = await Runner.run(starting_agent=agent, input=query)
        return [{"page_id": page.page_id} for page in result.final_output.results]

    async def get_confluence_pages(self, pages_list: List[Dict]) -> str:
        """Retrieve content for multiple confluence pages"""
        if not pages_list:
            raise ValueError("pages_list must be a non-empty list")

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

                            filename = os.path.join(output_dir, f'{page_id}.md')
                            with open(filename, 'w', encoding='utf-8') as md_file:
                                md_file.write(content)
                                print(f"Created file: {filename}")

                        except Exception as e:
                            print(f"Error processing page {page_id}: {str(e)}")

            return f"Files created successfully in {output_dir} folder"

        except Exception as e:
            raise RuntimeError(f"Failed to process confluence pages: {str(e)}")

    async def run_confluence_workflow(self, query: str) -> str:
        """Run complete confluence workflow - search and retrieve pages"""
        pages = await self.search_confluence(query)
        if pages:
            return await self.get_confluence_pages(pages)
        return "No pages found for the given query"

    async def search_github_repos(self, query: str) -> List[Dict]:
        """Search for GitHub repositories matching the query"""
        if not self.agents:
            raise RuntimeError("Servers not initialized. Call initialize_servers() first")

        agent = self.agents.get_github_agent()
        result = await Runner.run(starting_agent=agent, input=query)
        return [{"owner": repo.owner, "repo": repo.repo, "path": repo.path, "branch": repo.branch} 
                for repo in result.final_output.results]

    async def get_github_files(self, repos_list: List[Dict]) -> str:
        """Retrieve content from GitHub repositories"""
        if not repos_list:
            raise ValueError("repos_list must be a non-empty list")

        output_dir = "mifos_github_mds"
        os.makedirs(output_dir, exist_ok=True)

        params = StdioServerParameters(
            command="npx",
            args=[
                "-y",
                "@modelcontextprotocol/server-github"
            ],
            env={
                "GITHUB_PERSONAL_ACCESS_TOKEN": os.getenv('GITHUB_PERSONAL_ACCESS_TOKEN')
            }
        )

        try:
            async with stdio_client(params) as streams:
                async with ClientSession(streams[0], streams[1]) as session:
                    await session.initialize()

                    for repo in repos_list:
                        try:
                            owner = repo.get("owner")
                            repo_name = repo.get("repo")
                            path = repo.get("path")
                            branch = repo.get("branch")

                            if not all([owner, repo_name, path, branch]):
                                print(f"Skipping invalid repo entry: {repo}")
                                continue

                            result = await session.call_tool("get_file_contents", {
                                "owner": owner,
                                "repo": repo_name,
                                "path": path,
                                "branch": branch
                            })

                            text_dict = eval(result.content[0].text)
                            content = text_dict["content"]
                            
                            # Create filename with repo name
                            base_filename = os.path.basename(path)
                            filename = os.path.join(output_dir, f'{repo_name}_{base_filename}')
                            with open(filename, 'w', encoding='utf-8') as md_file:
                                md_file.write(content)
                                print(f"Created file: {filename}")

                        except Exception as e:
                            print(f"Error processing repo {repo_name}: {str(e)}")

            return f"Files created successfully in {output_dir} folder"

        except Exception as e:
            raise RuntimeError(f"Failed to process GitHub repositories: {str(e)}")

    async def run_github_workflow(self, query: str) -> str:
        """Run complete GitHub workflow - search and retrieve repository files"""
        repos = await self.search_github_repos(query)
        if repos:
            return await self.get_github_files(repos)
        return "No repositories found for the given query"
