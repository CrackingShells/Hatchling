from mcp_utils.hatch_mcp import HatchMCP

# Initialize MCP server with metadata
hatch_mcp = HatchMCP("version_dep_pkg",
                origin_citation="Version dependency package for testing",
                mcp_citation="Version dependency package MCP implementation")

@hatch_mcp.server.tool()
def version_specific_function(param: str) -> str:
    """Function that depends on a specific version of base_pkg_1.
    
    Args:
        param: Input parameter
        
    Returns:
        str: Processed result
    """
    hatch_mcp.logger.info(f"Version specific function called with param: {param}")
    # In a real implementation, this would call functions from a specific version of base_pkg_1
    return f"Version dependency package processed: {param} (depends on base_pkg_1 version 1.0.0)"

if __name__ == "__main__":
    hatch_mcp.logger.info("Starting MCP server")
    hatch_mcp.server.run()
