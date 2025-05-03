from mcp_utils.hatch_mcp import HatchMCP

# Initialize MCP server with metadata
hatch_mcp = HatchMCP("circular_dep_pkg_1",
                origin_citation="Circular dependency package 1 for testing",
                mcp_citation="Circular dependency package 1 MCP implementation")

@hatch_mcp.server.tool()
def circular_function_1(param: str) -> str:
    """Function from circular dependency package 1.
    
    Args:
        param: Input parameter
        
    Returns:
        str: Processed result
    """
    hatch_mcp.logger.info(f"Circular function 1 called with param: {param}")
    # In a real implementation, this would call functions from circular_dep_pkg_2
    return f"Circular dependency package 1 processed: {param} (depends on circular_dep_pkg_2)"

if __name__ == "__main__":
    hatch_mcp.logger.info("Starting MCP server")
    hatch_mcp.server.run()
