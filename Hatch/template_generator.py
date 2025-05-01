"""
Template Generator for Hatch packages.

This module contains functions to generate template files for Hatch MCP server packages.
Each function generates a specific file for the package template.
"""
import logging

logger = logging.getLogger("hatch.template_generator")

def generate_init_py():
    """Generate the __init__.py file content for a template package."""
    return "# Hatch package initialization\n"

def generate_server_py(package_name: str):
    """
    Generate the server.py file content for a template package.
    
    Args:
        package_name: Name of the package
        
    Returns:
        str: Content for server.py
    """
    return f"""import logging
from mcp_utils.hatch_mcp import HatchMCP

# Initialize MCP server with metadata
hatch_mcp = HatchMCP("{package_name}",
                origin_citation="Origin citation for {package_name}",
                mcp_citation="MCP citation for {package_name}")

# Example tool function
@hatch_mcp.server.tool()
def example_tool(param: str) -> str:
    \"\"\"Example tool function.
    
    Args:
        param: Example parameter
        
    Returns:
        str: Example result
    \"\"\"
    hatch_mcp.logger.info(f"Example tool called with param: {{param}}")
    return f"Processed: {{param}}"

if __name__ == "__main__":
    hatch_mcp.logger.info("Starting MCP server")
    hatch_mcp.server.run()
"""

def generate_metadata_json(package_name: str, category: str = "", description: str = ""):
    """
    Generate the metadata JSON content for a template package.
    
    Args:
        package_name: Name of the package
        category: Package category
        description: Package description
        
    Returns:
        dict: Metadata dictionary
    """
    return {
        "name": package_name,
        "version": "0.1.0",
        "description": description,
        "category": category,
        "tags": [],
        "author": {
            "name": "Hatch User",
            "email": ""
        },
        "license": "MIT",
        "entry_point": "server.py",
        "tools": [
            {
                "name": "example_tool",
                "description": "Example tool function"
            }
        ],
        "citations": {
            "origin": f"Origin citation for {package_name}",
            "mcp": f"MCP citation for {package_name}"
        }
    }

def generate_readme_md(package_name: str, description: str = ""):
    """
    Generate the README.md file content for a template package.
    
    Args:
        package_name: Name of the package
        description: Package description
        
    Returns:
        str: Content for README.md
    """
    return f"""# {package_name}

{description}

## Tools

- **example_tool**: Example tool function
"""
