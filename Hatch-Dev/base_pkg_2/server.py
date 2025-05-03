from mcp_utils.hatch_mcp import HatchMCP

# Initialize MCP server with metadata
hatch_mcp = HatchMCP("base_pkg_2",
                origin_citation="Base package 2 for testing with no dependencies",
                mcp_citation="Base package 2 MCP implementation")

@hatch_mcp.server.tool()
def base_function_alt(param: str) -> str:
    """Alternative base function for testing.
    
    Args:
        param: Input parameter
        
    Returns:
        str: Processed result
    """
    hatch_mcp.logger.info(f"Base function alt called with param: {param}")
    return f"Base package 2 processed: {param}"

if __name__ == "__main__":
    hatch_mcp.logger.info("Starting MCP server")
    hatch_mcp.server.run()
