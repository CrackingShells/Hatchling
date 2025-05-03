from mcp_utils.hatch_mcp import HatchMCP

hatch_mcp = HatchMCP("ArithmeticTools",
                origin_citation="Jacopin Eliott, \"Origin: Example MCP Server for Hatch!\", April 2025",
                mcp_citation="Jacopin Eliott, \"MCP: Example Arithmetic Tools for Hatch!\", April 2025")

@hatch_mcp.server.tool()
def add(a: float, b: float) -> float:
    """Add two numbers together.
    
    Args:
        a (float): First number.
        b (float): Second number.
        
    Returns:
        float: Sum of a and b.
    """
    hatch_mcp.logger.info(f"Adding {a} + {b}")
    return a + b

@hatch_mcp.server.tool()
def subtract(a: float, b: float) -> float:
    """Subtract second number from first number.
    
    Args:
        a (float): First number.
        b (float): Second number.
        
    Returns:
        float: Difference (a - b).
    """
    hatch_mcp.logger.info(f"Subtracting {a} - {b}")
    return a - b

@hatch_mcp.server.tool()
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together.
    
    Args:
        a (float): First number.
        b (float): Second number.
        
    Returns:
        float: Product of a and b.
    """
    hatch_mcp.logger.info(f"Multiplying {a} * {b}")
    return a * b

@hatch_mcp.server.tool()
def divide(a: float, b: float) -> float:
    """Divide first number by second number.
    
    Args:
        a (float): First number (dividend).
        b (float): Second number (divisor).
        
    Returns:
        float: Quotient (a / b).
        
    Raises:
        ValueError: If the divisor (b) is zero.
    """
    if b == 0:
        hatch_mcp.logger.error("Division by zero attempted")
        raise ValueError("Cannot divide by zero")
    
    hatch_mcp.logger.info(f"Dividing {a} / {b}")
    return a / b

if __name__ == "__main__":
    hatch_mcp.logger.info("Starting MCP server with arithmetic tools")
    
    hatch_mcp.server.run()