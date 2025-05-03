from mcp_utils.hatch_mcp import HatchMCP

# Initialize MCP server with metadata
hatch_mcp = HatchMCP("complex_dep_pkg",
                origin_citation="Complex dependency package for testing",
                mcp_citation="Complex dependency package MCP implementation")

@hatch_mcp.server.tool()
def complex_function(param1: str, param2: str) -> str:
    """Complex function that depends on both base packages.
    
    Args:
        param1: First input parameter (for base_pkg_1)
        param2: Second input parameter (for base_pkg_2)
        
    Returns:
        str: Combined processed result
    """
    hatch_mcp.logger.info(f"Complex function called with params: {param1}, {param2}")
    # In a real implementation, this would call functions from both base packages
    return f"Complex dependency package processed: {param1} + {param2} (depends on base_pkg_1 and base_pkg_2)"

if __name__ == "__main__":
    hatch_mcp.logger.info("Starting MCP server")
    hatch_mcp.server.run()
