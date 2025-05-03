import requests  # This would be the Python dependency
from mcp_utils.hatch_mcp import HatchMCP

# Initialize MCP server with metadata
hatch_mcp = HatchMCP("python_dep_pkg",
                origin_citation="Python dependency package for testing",
                mcp_citation="Python dependency package MCP implementation")

@hatch_mcp.server.tool()
def python_dep_function(url: str) -> str:
    """Function that depends on an external Python package (requests).
    
    Args:
        url: URL to fetch
        
    Returns:
        str: Status code from the request
    """
    hatch_mcp.logger.info(f"Python dependency function called with URL: {url}")
    # This actually uses the external Python dependency
    try:
        response = requests.get(url, timeout=10)
        return f"Python dependency package processed: URL {url} returned status code {response.status_code}"
    except Exception as e:
        return f"Python dependency package error: {str(e)}"

if __name__ == "__main__":
    hatch_mcp.logger.info("Starting MCP server")
    hatch_mcp.server.run()
