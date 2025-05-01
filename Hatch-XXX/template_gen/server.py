import logging
from mcp_utils.hatch_mcp import HatchMCP

# Initialize MCP server with metadata
hatch_mcp = HatchMCP("template_gen",
                origin_citation="Origin citation for template_gen",
                mcp_citation="MCP citation for template_gen")

# Example tool function
@hatch_mcp.server.tool()
def example_tool(param: str) -> str:
    """Example tool function.
    
    Args:
        param: Example parameter
        
    Returns:
        str: Example result
    """
    hatch_mcp.logger.info(f"Example tool called with param: {param}")
    return f"Processed: {param}"

if __name__ == "__main__":
    hatch_mcp.logger.info("Starting MCP server")
    hatch_mcp.server.run()
