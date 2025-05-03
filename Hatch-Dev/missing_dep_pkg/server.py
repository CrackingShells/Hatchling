from mcp_utils.hatch_mcp import HatchMCP

# Initialize MCP server with metadata
hatch_mcp = HatchMCP("missing_dep_pkg",
                origin_citation="Missing dependency package for testing",
                mcp_citation="Missing dependency package MCP implementation")

@hatch_mcp.server.tool()
def missing_dep_function(param: str) -> str:
    """Function that depends on a non-existent package.
    
    Args:
        param: Input parameter
        
    Returns:
        str: Processed result
    """
    hatch_mcp.logger.info(f"Missing dependency function called with param: {param}")
    # In a real implementation, this would try to call functions from non_existent_pkg
    return f"Missing dependency package processed: {param} (depends on non_existent_pkg)"

if __name__ == "__main__":
    hatch_mcp.logger.info("Starting MCP server")
    hatch_mcp.server.run()
