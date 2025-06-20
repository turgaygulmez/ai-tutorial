#!/usr/bin/env node
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ErrorCode,
  ListToolsRequestSchema,
  McpError,
} from "@modelcontextprotocol/sdk/types.js";
import { PEOPLE } from "./mock";

class PeopleServer {
  private server: Server;

  constructor() {
    console.error("[Setup] Initializing People MCP server...");

    this.server = new Server(
      {
        name: "people-mcp-server",
        version: "0.1.0",
      },
      {
        capabilities: {
          tools: {},
        },
      }
    );

    this.setupToolHandlers();

    this.server.onerror = (error) => console.error("[Error]", error);

    process.on("SIGINT", async () => {
      await this.server.close();
      process.exit(0);
    });
  }

  private setupToolHandlers() {
    this.server.setRequestHandler(ListToolsRequestSchema, async () => ({
      tools: [
        {
          name: "get_personal_data",
          description: "Get personal data for a person",
          inputSchema: {
            type: "object",
            properties: {
              fullName: {
                type: "string",
                description: "Full name of the person",
              },
            },
            required: ["fullName"],
          },
        },
      ],
    }));

    this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
      try {
        if (!["get_personal_data"].includes(request.params.name)) {
          throw new McpError(
            ErrorCode.MethodNotFound,
            `Unknown tool: ${request.params.name}`
          );
        }

        const args = request.params.arguments as {
          fullName: string;
        };

        if (!args.fullName) {
          throw new McpError(
            ErrorCode.InvalidParams,
            "Missing required parameter: fullName"
          );
        }
        const person = PEOPLE.find(
          (x) => x.fullName.toLowerCase() === args.fullName.toLowerCase()
        );

        return {
          content: [
            {
              type: "text",
              text: JSON.stringify(person, null, 2),
            },
          ],
        };
      } catch (error: unknown) {
        if (error instanceof Error) {
          console.error("[Error] Failed to fetch data:", error);
          throw new McpError(
            ErrorCode.InternalError,
            `Failed to fetch data: ${error.message}`
          );
        }
        throw error;
      }
    });
  }

  async run() {
    const transport = new StdioServerTransport();
    await this.server.connect(transport);
    console.error("People MCP server running on stdio");
  }
}

const server = new PeopleServer();
server.run().catch(console.error);
