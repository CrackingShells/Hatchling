from mcp_utils.hatch_mcp import HatchMCP

# Initialize MCP server with metadata
hatch_mcp = HatchMCP("circular_dep_pkg_2",
                origin_citation="Circular dependency package 2 for testing",
                mcp_citation="Circular dependency package 2 MCP implementation")

@hatch_mcp.server.tool()
def circular_function_2(param: str) -> str:
    """Function from circular dependency package 2.
    
    Args:
        param: Input parameter
        
    Returns:
        str: Processed result
    """
    hatch_mcp.logger.info(f"Circular function 2 called with param: {param}")
    # In a real implementation, this would call functions from circular_dep_pkg_1
    return f"Circular dependency package 2 processed: {param} (depends on circular_dep_pkg_1)"

if __name__ == "__main__":
    hatch_mcp.logger.info("Starting MCP server")
    hatch_mcp.server.run()
