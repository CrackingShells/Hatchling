import os
import asyncio
import logging
from typing import Dict, Any, Optional
import uuid
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from hatchling.core.logging.logging_manager import logging_manager


class MCPClient:
    """Client for MCP servers that manages connections and tool execution."""
    def __init__(self):
        """Initialize the MCP client."""
        self.client_id = str(uuid.uuid4())
        self.session: Optional[ClientSession] = None
        self.exit_stack = None  # Created in connection manager task
        self.connected = False
        self.tools = {}
        self.server_path = None
        self.read = None
        self.write = None
        
        # Connection manager task and queue
        self._operation_queue = asyncio.Queue()
        self._manager_task = None
        self._manager_lock = asyncio.Lock()
        self._connection_task_id = None  # Track the ID of the task that manages connections
        
        # Add connection monitoring
        self._heartbeat_task = None
        self._reconnection_attempts = 0
        self.MAX_RECONNECTION_ATTEMPTS = 3
        self.RECONNECTION_DELAY = 2  # seconds
        
        # Get a debug log session from the LoggingManager
        self.logger = logging_manager.get_session(self.__class__.__name__,
                                  formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    async def connect(self, server_path: str) -> bool:
        """Connect to an MCP server via stdio.
        
        Args:
            server_path (str): Path to the server script (.py file).
            
        Returns:
            bool: True if connection was successful, False otherwise.
        """
        # Start the connection manager if not running
        try:
            await self._start_connection_manager()
            
            # Create a future to get the result
            future = asyncio.Future()
            await self._operation_queue.put(("connect", [server_path], future))
            
            # Wait for the operation to complete
            return await asyncio.wait_for(future, timeout=30)
        except asyncio.TimeoutError:
            self.logger.error(f"Connection attempt to {server_path} timed out after 30 seconds")
            return False
        except Exception as e:
            self.logger.error(f"Error in connect operation: {e}")
            return False
    
    def _start_heartbeat(self):
        """Start a background task to periodically check connection health."""
        if self._heartbeat_task is None:
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
    async def _heartbeat_loop(self):
        """Periodically check if the connection is still alive."""
        try:
            while self.connected:
                await asyncio.sleep(30)  # Check every 30 seconds
                if not self.connected:
                    break
                    
                try:
                    # Try a lightweight operation to check connection
                    if self.session:
                        await self.session.send_ping()
                        self.logger.debug("Connection heartbeat: OK")
                except Exception as e:
                    self.logger.warning(f"Connection heartbeat failed: {e}")
                    # Mark as disconnected without trying to reconnect
                    # This avoids task/context issues
                    self.logger.warning("Connection marked as failed - client needs to be reconnected")
                    self.connected = False
                    break
                    
        except asyncio.CancelledError:
            self.logger.debug("Heartbeat task cancelled")
        except Exception as e:
            self.logger.error(f"Error in heartbeat task: {e}")
        finally:
            self.logger.debug("Heartbeat task stopped")
            self._heartbeat_task = None # _cleanup_connection has been replaced by _internal_cleanup in the connection manager
    
    async def disconnect(self):
        """Disconnect from the MCP server and clean up resources."""
        if not self.connected:
            return
            
        if not self._manager_task or self._manager_task.done():
            # If there's no manager task running, mark as disconnected directly
            self.connected = False
            self.session = None
            self.exit_stack = None
            self.tools = {}
            return
        
        try:
            # Create a future to get the result
            future = asyncio.Future()
            await self._operation_queue.put(("disconnect", [], future))
            
            # Wait for the operation to complete with timeout
            await asyncio.wait_for(future, timeout=10)
        except asyncio.TimeoutError:
            self.logger.error("Disconnect operation timed out after 10 seconds")
            # Even if disconnect fails, mark as disconnected
            self.connected = False
        except Exception as e:
            self.logger.error(f"Error in disconnect operation: {e}")
            # Even if disconnect fails, mark as disconnected
            self.connected = False
    async def get_citations(self) -> Dict[str, str]:
        """Get citations from the MCP server.
        
        Returns:
            Dict[str, str]: Dictionary with origin and MCP citations.
        """
        if not self.connected:
            raise ConnectionError("Not connected to MCP server")
            
        # Create a future to get the result
        future = asyncio.Future()
        await self._operation_queue.put(("get_citations", [], future))
        
        # Wait for the operation to complete
        try:
            return await future
        except Exception as e:
            self.logger.error(f"Error in get_citations operation: {e}")
            raise
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute an MCP tool by name with given arguments.
        
        Args:
            tool_name (str): Name of the tool to execute.
            arguments (Dict[str, Any): Arguments to pass to the tool.
            
        Returns:
            Any: Result of the tool execution.
            
        Raises:
            ConnectionError: If not connected to MCP server.
            ValueError: If the tool is not found.
            TimeoutError: If the tool execution times out.
            Exception: For any other errors during execution.
        """
        if not self.connected or not self.session:
            raise ConnectionError("Not connected to MCP server")
            
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not found")
        
        try:            
            # Execute the tool through the connection manager task
            future = asyncio.Future()
            await self._operation_queue.put(("execute_tool", [tool_name, arguments], future))
            
            # Wait for the result
            return await future
                
        except Exception as e:
            self.logger.error(f"Error executing tool {tool_name}: {str(e)}")
            raise

    async def _start_connection_manager(self):
        """Start the dedicated connection manager task if not already running."""
        async with self._manager_lock:
            # If there's a task but it's done, clean it up first
            if self._manager_task and self._manager_task.done():
                # Check if the task had an exception
                try:
                    exc = self._manager_task.exception()
                    if exc:
                        self.logger.error(f"Previous connection manager failed with: {exc}")
                except (asyncio.InvalidStateError, asyncio.CancelledError):
                    # Task was cancelled or is in an invalid state
                    pass
                
                self._manager_task = None
                
            # Create a new task if needed
            if self._manager_task is None:
                self.logger.debug("Starting connection manager task")
                self._manager_task = asyncio.create_task(self._connection_manager_loop())
                # Name the task for better debugging
                self._manager_task.set_name(f"mcp_connection_manager_{self.client_id[:8]}")

    async def _connection_manager_loop(self):
        """A dedicated task that handles all connection and disconnection operations.
        
        This ensures all operations with exit stacks and cancel scopes happen in the same task context.
        """
        # Store the task ID to track task context
        self._connection_task_id = id(asyncio.current_task())
        self.logger.debug(f"Connection manager task started with ID: {self._connection_task_id}")
        
        try:
            while True:
                # Get the next operation from the queue
                operation, args, future = await self._operation_queue.get()
                
                try:
                    self.logger.debug(f"Processing operation: {operation}")
                    if operation == "connect":
                        server_path = args[0]
                        result = await self._internal_connect(server_path)
                        future.set_result(result)
                    elif operation == "disconnect":
                        await self._internal_disconnect()
                        future.set_result(None)
                    elif operation == "execute_tool":
                        tool_name, arguments = args
                        result = await self._internal_execute_tool(tool_name, arguments)
                        future.set_result(result)
                    elif operation == "get_citations":
                        result = await self._internal_get_citations()
                        future.set_result(result)
                    else:
                        self.logger.warning(f"Unknown operation: {operation}")
                        future.set_exception(ValueError(f"Unknown operation: {operation}"))
                except Exception as e:
                    self.logger.error(f"Error processing operation {operation}: {e}")
                    future.set_exception(e)
                finally:
                    self._operation_queue.task_done()
        except asyncio.CancelledError:
            self.logger.debug(f"Connection manager task {self._connection_task_id} cancelled")
        except Exception as e:
            self.logger.error(f"Error in connection manager loop: {e}")
        finally:
            self.logger.debug(f"Connection manager task {self._connection_task_id} stopped")
            self._connection_task_id = None

    async def _internal_connect(self, server_path: str) -> bool:
        """Internal connect method that runs in the connection manager task.
        
        Args:
            server_path (str): Path to the server script.
            
        Returns:
            bool: True if connection was successful.
        """
        # Clean up any existing connection first
        await self._internal_cleanup()
            
        self.server_path = server_path
        # Create a fresh exit stack for this connection
        self.exit_stack = AsyncExitStack()
        
        # Create server parameters for stdio connection
        server_params = StdioServerParameters(
            command="python",
            args=[server_path],
            env=os.environ.copy(),
        )
        
        self.logger.debug(f"Connecting to MCP server: {server_path}")
        try:
            # Follow the exact sequence from the working example
            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
            self.read, self.write = stdio_transport
            session = await self.exit_stack.enter_async_context(ClientSession(self.read, self.write))
            await session.initialize()
            self.session = session
            self.connected = True
            
            # List available tools
            response = await self.session.list_tools()
            tools = response.tools
            
            # Store tools in a dictionary for easy access
            for tool in tools:
                self.tools[tool.name] = tool
            
            self.logger.info(f"Connected to MCP server: {server_path}")
            self.logger.info(f"Discovered {len(self.tools)} tools: {', '.join(self.tools.keys())}")

            # List available citations
            citations = await self._internal_get_citations()
            self.logger.info("Tool Origin Citation: " + citations["origin"])
            self.logger.info("MCP Implementation Citation: " + citations["mcp"])
            
            # Start heartbeat task to monitor connection
            self._start_heartbeat()
            
            return True
            
        except asyncio.CancelledError:
            self.logger.warning("Connection attempt cancelled")
            await self._internal_cleanup()
            raise
        except Exception as e:
            self.logger.error(f"Failed to connect to MCP server at {server_path}: {str(e)}")
            await self._internal_cleanup()
            return False
            
    async def _internal_disconnect(self):
        """Internal disconnect method that runs in the connection manager task."""
        if not self.connected:
            return
            
        self.logger.debug(f"Disconnecting from MCP server: {self.server_path} in task {self._connection_task_id}")
        
        # Cancel heartbeat task first
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                # Wait for task to fully cancel (but don't wait indefinitely)
                await asyncio.wait_for(asyncio.shield(self._heartbeat_task), timeout=2)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                # It's okay if the task times out or is cancelled
                pass
            finally:
                self._heartbeat_task = None
        
        # Clean up resources
        try:
            await self._internal_cleanup()
        except Exception as e:
            self.logger.error(f"Error during disconnection cleanup: {e}")
            # Even if cleanup fails, continue with state reset
        
        # Reset additional state
        self.tools = {}
        self._reconnection_attempts = 0
        
        self.logger.info(f"Disconnected from MCP server: {self.server_path}")
    
    async def _internal_cleanup(self):
        """Internal cleanup method that runs in the connection manager task."""
        self.logger.debug(f"Starting connection cleanup in task {self._connection_task_id}")
        
        try:
            if self.exit_stack:
                self.logger.debug("Closing exit stack from the same task that created it")
                await self.exit_stack.aclose()
        except asyncio.CancelledError:
            self.logger.warning("Cleanup interrupted by cancellation")
            raise  # Re-raise to allow proper handling
        except Exception as e:
            self.logger.error(f"Error during connection cleanup: {e}")
        finally:
            # Always reset these regardless of success/failure
            self.exit_stack = None
            self.session = None
            self.read = None
            self.write = None
            self.connected = False
            self.logger.debug("Connection state reset completed")
    
    async def _internal_execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Internal execute tool method that runs in the connection manager task."""
        if not self.connected or not self.session:
            raise ConnectionError("Not connected to MCP server")
            
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not found")
        
        try:
            # Execute the tool with timeout
            result = await asyncio.wait_for(
                self.session.call_tool(name=tool_name, arguments=arguments),
                timeout=30  # 30 second timeout
            )
            
            # Extract the result value from the response object if needed
            if hasattr(result, 'result'):
                return result.result
            else:
                return result
                
        except asyncio.TimeoutError:
            self.logger.error(f"Tool execution timed out: {tool_name}")
            raise TimeoutError(f"Execution of tool {tool_name} timed out after 30 seconds")
            
        except Exception as e:
            self.logger.error(f"Error executing tool {tool_name}: {str(e)}")
            raise

    async def _internal_get_citations(self) -> Dict[str, str]:
        """Internal get citations method that runs in the connection manager task."""
        if not self.connected or not self.session:
            raise ConnectionError("Not connected to MCP server")
            
        citations = {
            "server_name": "None",
            "origin": "Citation not available",
            "mcp": "Citation not available"
        }

        try:
            # Extract server name from the path server uri
            try:
                server_name_uri = f"name://{self.server_path[1:]}"
                server_name_response = await self.session.read_resource(uri=server_name_uri)
                if server_name_response and server_name_response.contents:
                    citations["server_name"] = server_name_response.contents[0].text
                    self.logger.debug(f"Retrieved server name from {server_name_uri}: {citations['server_name']}")
            except Exception as e:
                self.logger.error(f"Failed to get server name: {e}")
                
            # Try to read origin citation
            try:
                origin_uri = f"citation://origin/{citations['server_name']}"
                origin_response = await self.session.read_resource(uri=origin_uri)
                if origin_response and origin_response.contents:
                    citations["origin"] = origin_response.contents[0].text
                    self.logger.debug(f"Retrieved origin citation from {origin_uri}")
            except Exception as e:
                self.logger.error(f"Failed to get origin citation: {e}")
            
            # Try to read MCP citation
            try:
                mcp_uri = f"citation://mcp/{citations['server_name']}"
                mcp_response = await self.session.read_resource(uri=mcp_uri)
                if mcp_response and mcp_response.contents:
                    citations["mcp"] = mcp_response.contents[0].text
                    self.logger.debug(f"Retrieved MCP citation from {mcp_uri}")
            except Exception as e:
                self.logger.error(f"Failed to get MCP citation: {e}")
        
        except Exception as e:
            self.logger.error(f"Error retrieving citations: {e}")
            
        return citations

    async def _stop_connection_manager(self):
        """Stop the connection manager task gracefully."""
        if self._manager_task and not self._manager_task.done():
            self.logger.debug("Stopping connection manager task")
            
            # Cancel the task
            self._manager_task.cancel()
            
            try:
                # Wait for the task to finish with timeout
                await asyncio.wait_for(self._manager_task, timeout=5)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                self.logger.warning("Connection manager task cancellation timed out or was cancelled")
            except Exception as e:
                self.logger.error(f"Error stopping connection manager task: {e}")
                
            self._manager_task = None