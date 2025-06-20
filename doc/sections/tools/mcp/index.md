# MCP

MCP is an open protocol that standardizes how applications provide context to LLMs. Think of MCP like a USB-C port for AI applications. Just as USB-C provides a standardized way to connect your devices to various peripherals and accessories, MCP provides a standardized way to connect AI models to different data sources and tools.

[documentation](https://modelcontextprotocol.io/introduction)

List of available MCP servers

[servers](https://mcpmarket.com/server)


## Create your own MCP

Create a MCP server using @modelcontextprotocol/sdk

[example](./samples/basic/src/index.ts)

Once MCP is running, add it to your MCP list

```json
{
  "mcpServers": {
    "people": {
      "command": "node",
      "args": [
        "ABSOLUTE_PATH/MCP-sample/build/index.js"
      ]
    }
  }
}
```




