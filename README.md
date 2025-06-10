# Hatchling

![Hatchling Logo](./doc/resources/images/Logo/hatchling_wide_dark_bg_transparent.png)

Hatchling is an interactive CLI-based chat application that integrates local Large Language Models (LLMs) through [Ollama](https://ollama.ai/) with the [Model Context Protocol](https://github.com/modelcontextprotocol) (MCP) for tool calling capabilities. It is meant to be the frontend for using all MCP servers in Hatch!

## Project Update Summary

**May 27, 2025**: First release of the Hatch package manager ecosystem! 🎉
- The Hatch package manager is now **fully integrated into Hatchling** with built-in commands
- The package architecture for dynamic loading and switching of MCP servers is now complete
- [Related repositories](#related-repositories) established to support the pipeline.

## Features

- Interactive CLI-based chat interface
- Integration with Ollama API for local LLM support
- Ollama tool calling to MCP tools
- Tool execution wrapping to babysit LLMs into doing longer tool calling chains to do more work
- Appropriate citation of the source software wrapped in the MCP server whenever the LLM uses them

## Roadmap

- Support for vanilla MCP servers syntax (no wrapping in `HatchMCP`)
- Launching **Hatch! Biology** for hosting MCP servers providing access to well-established software and methods such as BLAST, UniProt (and other database) queries, PubMed articles, and such... All with citations!
- Customize LLMs system prompts, reference past messages, be in control of context history
- GUI for the chat and all management of the MCP servers
- User-defined tool chains

## Prerequisites

- [Docker Desktop](https://docs.docker.com/desktop/) installed and configured with WSL2 (Windows) or running properly on your system (macOS/Linux)
- Recommended: GPU support configured for better performance with LLMs

## Installation & Running

Hatchling is designed to run with Docker for a consistent experience across platforms:

1. **Setup Docker and Ollama**:
   - Follow the [detailed Docker setup instructions](./doc/articles/docker-setup.md) to set up Docker Desktop and Ollama with GPU support (if available)

2. **Run Hatchling**:
   ```bash
   # From the docker directory in your project
   export USER_ID=$(id -u)
   export GROUP_ID=$(id -g)
   docker-compose run --rm hatchling
   ```

For complete installation and setup details, including GPU configuration, see our [Docker Setup Guide](./doc/articles/docker-setup.md).

## Configuration

Configuration is managed through environment variables or a `.env` file in the `docker` directory:

| Variable | Description | Default |
|----------|-------------|---------|
| `OLLAMA_HOST_API` | URL for the Ollama API | `http://localhost:11434/api` |
| `OLLAMA_MODEL` | Default LLM model to use | `llama3.2` |
| `HATCH_HOST_CACHE_DIR` | Directory where Hatch environments and cache will be stored on the host machine | `./.hatch` |
| `HATCH_LOCAL_PACKAGE_DIR` | Directory where local packages are stored on the host machine to be accessible in the container | `../../Hatch_Pkg_Dev` |
| `NETWORK_MODE` | Docker network mode | `host` (for Linux) |
| `LOG_LEVEL` | The default log level at start up | `INFO` |

## Usage

### Running with Docker

```bash
# Basic usage
docker-compose -f docker/docker-compose.yml run --rm hatchling

# Specify a different model
docker-compose -f docker/docker-compose.yml run --rm -e OLLAMA_MODEL=llama2 hatchling

# Start with different environment directories
docker-compose -f docker/docker-compose.yml run --rm -e HATCH_HOST_CACHE_DIR=./my_hatch_cache -e HATCH_LOCAL_PACKAGE_DIR=../my_packages hatchling
```

### Chat Commands

The following commands are available during chat:

#### Basic Commands

| Command | Description | Arguments | Example |
|---------|-------------|----------|---------|
| `help` | Display help for available commands | None | `help` |
| `clear` | Clear the chat history | None | `clear` |
| `enable_tools` | Enable MCP tools | None | `enable_tools` |
| `disable_tools` | Disable MCP tools | None | `disable_tools` |
| `show_logs` | Display session logs | `[n]` - Optional number of log entries to show | `show_logs` or `show_logs 10` |
| `set_log_level` | Change log level | `<level>` - Log level (debug, info, warning, error, critical) | `set_log_level debug` |
| `set_max_tool_call_iterations` | Set maximum tool call iterations | `<n>` - Maximum iterations | `set_max_tool_call_iterations 10` |
| `set_max_working_time` | Set maximum working time in seconds | `<seconds>` - Maximum time | `set_max_working_time 60` |
| `exit` or `quit` | End the chat session | None | `exit` |

#### Hatch Environment Management

| Command | Description | Arguments | Example |
|---------|-------------|----------|---------|
| `hatch:env:list` | List all available Hatch environments | None | `hatch:env:list` |
| `hatch:env:create` | Create a new Hatch environment | `<name>` - Environment name<br>`--description <description>` - Environment description | `hatch:env:create my-env --description "For biology tools"` |
| `hatch:env:remove` | Remove a Hatch environment | `<name>` - Environment name | `hatch:env:remove my-env` |
| `hatch:env:current` | Show the current Hatch environment | None | `hatch:env:current` |
| `hatch:env:use` | Set the current Hatch environment | `<name>` - Environment name | `hatch:env:use my-env` |

#### Hatch Package Management

| Command | Description | Arguments | Example |
|---------|-------------|----------|---------|
| `hatch:pkg:add` | Add a package to an environment | `<package_path_or_name>` - Path or name of package<br>`--env <env_name>` - Environment name<br>`--version <version>` - Package version | `hatch:pkg:add ./my-package --env my-env` |
| `hatch:pkg:remove` | Remove a package from an environment | `<package_name>` - Name of package to remove<br>`--env <env_name>` - Environment name | `hatch:pkg:remove my-package --env my-env` |
| `hatch:pkg:list` | List packages in an environment | `--env <env_name>` - Environment name | `hatch:pkg:list --env my-env` |
| `hatch:create` | Create a new package template | `<name>` - Package name<br>`--dir <dir>` - Target directory<br>`--description <description>` - Package description | `hatch:create my-package --description "My MCP package"` |
| `hatch:validate` | Validate a package | `<package_dir>` - Path to package directory | `hatch:validate ./my-package` |

### Example Usage

```
Starting interactive chat with mistral-small3.1

=== Chat Commands ===
Type 'help' for this help message

Type 'clear' - Clear the chat history
Type 'disable_tools' - Disable MCP tools
Type 'enable_tools' - Enable MCP tools
Type 'exit' - End the chat session
Type 'hatch:create' - Create a new package template. Usage: hatch:create <name> [--dir <dir>] [--category <category>] [--description <description>]
Type 'hatch:env:create' - Create a new Hatch environment. Usage: hatch:env:create <name> [--description <description>]
Type 'hatch:env:current' - Show the current Hatch environment
Type 'hatch:env:list' - List all available Hatch environments
Type 'hatch:env:remove' - Remove a Hatch environment. Usage: hatch:env:remove <name>
Type 'hatch:env:use' - Set the current Hatch environment. Usage: hatch:env:use <name>
Type 'hatch:pkg:add' - Add a package to an environment. Usage: hatch:pkg:add <package_path_or_name> [--env <env_name>] [--version <version>]
Type 'hatch:pkg:list' - List packages in an environment. Usage: hatch:pkg:list [--env <env_name>]
Type 'hatch:pkg:remove' - Remove a package from an environment. Usage: hatch:pkg:remove <package_name> [--env <env_name>]
Type 'hatch:validate' - Validate a package. Usage: hatch:validate <package_dir>
Type 'help' - Display help for available commands
Type 'set_log_level' - Change log level (debug, info, warning, error, critical). Usage: set_log_level <level>
Type 'set_max_tool_call_iterations' - Set max tool call iterations. Usage: set_max_tool_call_iterations <n>
Type 'set_max_working_time' - Set max working time in seconds. Usage: set_max_working_time <seconds>
Type 'show_logs' - Display session logs. Usage: show_logs [n]
======================

[Tools disabled] You: enable_tools
Tools enabled successfully!

[Tools enabled] You: What is 15 * 7?
Assistant: Let me calculate 15 * 7 for you. #<-- This line might be different for you

## DEBUG/INFO level log ##

[Using tool: multiply with arguments: {'a': 15.0, 'b': 7.0}]

## DEBUG/INFO level log ##

[Tool result: 105.0]

## DEBUG/INFO level log ##

Final response based on tool results:
The result of 15 multiplied by 7 is 105. #<-- This line might be different for you

<-- vvv This section below might be different for you vvv -->

### Citations
- ArithmeticTools
  Origin: Jacopin Eliott, "Origin: Example MCP Server for Hatch!", April 2025
  Implementation: Jacopin Eliott, "MCP: Example Arithmetic Tools for Hatch!", April 2025
```

## Extending with your MCP Servers

You can extend Hatchling with custom MCP tools by creating Hatch packages:

1. Create a new package using the template generator:

   ```bash
   docker-compose -f docker/docker-compose.yml run --rm hatchling
   ```

   Then use the command: `hatch:create <name> --description "Your description here"`

2. Define a new Hatch MCP server in your package:

   ```Python
   from mcp.server.fastmcp import FastMCP
   hatch_mcp = HatchMCP("NAME",
                   origin_citation="SOFTWARE ORIGIN",
                   mcp_citation="CREDITS for the MCP SERVER")
   ```

3. Define your MCP tools using the `@hatch_mcp.tool()` decorator

4. Add your package to your environment with:

   ```
   hatch:pkg:add /path/to/your/package
   ```

5. Enable tools during chat with the `enable_tools` command

### Example Custom Tool

An example of MCP server:

```python
from mcp.server.fastmcp import FastMCP
hatch_mcp = HatchMCP("NAME",
              origin_citation="SOFTWARE ORIGIN",
              mcp_citation="CREDITS for the MCP SERVER")

@hatch_mcp.tool()
def my_custom_tool(param1: str, param2: int) -> str:
    """Description of what your tool does.
    
    Args:
        param1 (str): First parameter description.
        param2 (int): Second parameter description.
        
    Returns:
        str: Description of the return value.
    """
    hatch_mcp.logger.info(f"Custom tool called with {param1} and {param2}")
    return f"Processed {param1} with value {param2}"

if __name__ == "__main__":
    hatch_mcp.run()
```

Each package should include a `hatch_metadata.json` file that follows the schema defined in the Hatch-Schemas repository.

## Development

### Project Structure

- `app.py`: Main entry point
- `config/`: Configuration management
- `core/`: Core functionality
  - `chat/`: Chat system components
  - `llm/`: LLM integration logic
  - `logging/`: Logging system
- `mcp_utils/`: MCP tool utilities
- `ui/`: User interface components
- `docker/`: Docker configuration files

## Related Repositories

Hatchling is part of the larger Hatch ecosystem which includes:

- **[Hatch](https://github.com/CrackingShells/Hatch)**: The official package manager for the Hatch ecosystem, now fully integrated into Hatchling with built-in commands for MCP server management
  - Provides environment management for MCP server collections
  - Handles package installation from both local and registry sources
  - Template function to jump start new package development
- **[Hatch-Schemas](https://github.com/CrackingShells/Hatch-Schemas)**: Contains the JSON schemas for package metadata and validation
  - Includes schemas for both individual packages and the central registry
  - Provides versioned access to schemas via GitHub releases
  - Offers helper utilities for schema caching and updates
- **[Hatch-Validator](https://github.com/CrackingShells/Hatch-Validator)**: Validates packages against the schemas
  - Performs package validation against schema specifications
  - Resolves and validates package dependencies
  - Automatically fetches and manages schema versions
- **[Hatch-Registry](https://github.com/CrackingShells/Hatch-Registry)**: Package registry for Hatch packages
  - Maintains a centralized repository of available MCP server packages
  - Supports package versioning and dependency information
  - Provides search and discovery functionality
  - Ensures package integrity through metadata verification

These repositories work together to provide a comprehensive framework for creating, managing, and using MCP tools in Hatchling.
