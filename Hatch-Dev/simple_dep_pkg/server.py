from mcp_utils.hatch_mcp import HatchMCP

# Initialize MCP server with metadata
hatch_mcp = HatchMCP("simple_dep_pkg",
                origin_citation="Simple dependency package for testing",
                mcp_citation="Simple dependency package MCP implementation")

@hatch_mcp.server.tool()
def simple_wrapper(param: str) -> str:
    """Simple wrapper function that depends on base_pkg_1.
    
    Args:
        param: Input parameter
        
    Returns:
        str: Processed result
    """
    hatch_mcp.logger.info(f"Simple wrapper called with param: {param}")
    # In a real implementation, this would call functions from base_pkg_1
    return f"Simple dependency package processed: {param} (depends on base_pkg_1)"

if __name__ == "__main__":
    hatch_mcp.logger.info("Starting MCP server")
    hatch_mcp.server.run()
