from mcp import MCPClient

# Initialize the MCP client
client = MCPClient(server_url="http://your-mcp-server.com")

# Send context and interact with the LLM
response = client.send_context({"user_input": "Hello, how are you?"})
print(response)