from agents import Agent
from agents.mcp import MCPServer
from pydantic import BaseModel
from typing import List


class ConfluencePage(BaseModel):
    page_id: str
    url: str
    title: str


class ConfluencePageList(BaseModel):
    results: list[ConfluencePage]


class GitHubFileLocation(BaseModel):
    owner: str
    repo: str
    path: str
    branch: str


class FileLocatorList(BaseModel):
    results: List[GitHubFileLocation]


class MifosAgents:
    def __init__(self, confluence_server: MCPServer, github_server: MCPServer):
        self.confluence_server = confluence_server
        self.github_server = github_server

    def get_confluence_search_agent(self) -> Agent:
        return Agent(
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
            mcp_servers=[self.confluence_server],
            output_type=ConfluencePageList,
        )

    def get_github_agent(self) -> Agent:
        return Agent(
            name="OpenMF Branch-Aware Repository Finder",
            instructions="""**Execution Protocol**
                1. **Strict Tool Usage**
                   - ONLY use `search_repositories(query: str)`
                   - NEVER call file content tools
                
                2. **Branch Detection**
                   a. Extract `default_branch` from search results
                   b. Use exact branch name from API response
                   c. Handle common variants: main, master, development, etc.
                
                3. **Output Rules**
                   - Maximum 3 most relevant repos
                   - Required fields from search results:
                     - repository.name → repo
                     - repository.default_branch → branch
                   - Always set:
                     - owner: "openMF"
                
                **Example Output:**
                [
                  {
                    "owner": "openMF",
                    "repo": "mcp-mifosx",
                    "path": "CONTRIBUTING.md",
                    "branch": "master"  # Actual branch from search result
                  },
                  {
                    "owner": "openMF",
                    "repo": "fineract-advancly",
                    "path": "README.md",
                    "branch": "development" # Actual branch from search result
                  }
                ]""",
            mcp_servers=[self.github_server],
            output_type=FileLocatorList
        )
