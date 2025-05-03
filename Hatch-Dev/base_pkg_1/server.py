from mcp_utils.hatch_mcp import HatchMCP

# Initialize MCP server with metadata
hatch_mcp = HatchMCP("base_pkg_1",
                origin_citation="Base package 1 for testing with no dependencies",
                mcp_citation="Base package 1 MCP implementation")

@hatch_mcp.server.tool()
def base_function(param: str) -> str:
    """Basic function for testing.
    
    Args:
        param: Input parameter
        
    Returns:
        str: Processed result
    """
    hatch_mcp.logger.info(f"Base function called with param: {param}")
    return f"Base package 1 processed: {param}"

if __name__ == "__main__":
    hatch_mcp.logger.info("Starting MCP server")
    hatch_mcp.server.run()
