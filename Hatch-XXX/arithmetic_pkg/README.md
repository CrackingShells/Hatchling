# Arithmetic Package

The default MCP package that is being used to test all MCP tools integration in Hatch!
Of course, it doesn't do nothing! With this wonderful (and cute) package, you empower your LLMs with additions, subtractions, multiplications, and divisions. Ain't it great?!... Right?

## Tools

- **add**: Add two numbers together.
- **subtract**: Subtract one number from another
- **multiply**: Multiply two numbers together
- **divide**: Divide one number by another

## Usage Remarks

The tools themselves take only two parameters. Hence, when faced with calculations with more than two operands (e.g. 8*9+4, 7/9-7, (8-6)*7+25/3, and so one...), LLMs might not use the tools.

It was observed, however, that prompting with `use the tools step-by-step` in was useful in cooperation with the tool-chain-calling feature in _Hatchling_. In such cases, the LLM would start computing independant parts of the total calculations. Note that the LLMs were always observed to follow the correct operation priotities when doing serial calling.

