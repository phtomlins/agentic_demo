# Agentic Demo

A simple demonstration of an agentic AI system that can use tools to accomplish tasks.

## Overview

This demo showcases the core concepts of agentic AI:
- **Agent**: An autonomous entity that can reason and take actions
- **Tools**: Functions the agent can use to accomplish tasks
- **Memory**: The agent's ability to remember past actions
- **Orchestration**: Coordinating tool usage to achieve goals

## Features

- Simple, clean implementation without heavy dependencies
- Multiple example tools (calculator, weather lookup, text analyzer)
- Memory system to track agent actions
- Extensible architecture for adding new tools

## Installation

No special installation required! Just Python 3.7+

```bash
git clone https://github.com/phtomlins/agentic_demo.git
cd agentic_demo
```

## Usage

Run the demo:

```bash
python agent.py
```

### Using the Agent in Your Code

```python
from agent import AgenticAgent, Tool

# Define a custom tool
def my_tool(input_data):
    return f"Processed: {input_data}"

# Create tools
tools = [
    Tool("my_tool", "Description of what my tool does", my_tool)
]

# Create and use the agent
agent = AgenticAgent(tools)
result = agent.use_tool("my_tool", "Hello, Agent!")
print(result)
```

## Available Tools

1. **Calculator**: Perform arithmetic operations (add, subtract, multiply, divide)
2. **Weather**: Get weather information for cities (mock data)
3. **Text Analyzer**: Analyze text and get statistics (character count, word count, etc.)

## Example Output

```
AGENTIC AI DEMO
============================================================

Available tools:
- calculator: Perform arithmetic operations (add, subtract, multiply, divide)
- weather: Get weather information for a city
- text_analyzer: Analyze text and get statistics

Example 1: Calculate 15 + 27
Result: 42

Example 2: Check weather in Tokyo
Result: Weather in Tokyo: Rainy, 68°F

Example 3: Analyze sample text
Result: {
  "character_count": 60,
  "word_count": 12,
  "sentence_count": 2,
  "average_word_length": 3.58
}
```

## Architecture

The demo consists of:
- `AgenticAgent`: Main agent class that manages tools and memory
- `Tool`: Class representing a callable tool
- Example tool functions demonstrating different capabilities

## Extending the Demo

To add your own tools:

1. Create a function that does something useful
2. Wrap it in a `Tool` object
3. Add it to the agent's tool list

```python
def my_custom_tool(param1, param2):
    # Your logic here
    return result

custom_tool = Tool(
    name="custom_tool",
    description="What this tool does",
    func=my_custom_tool
)

agent = AgenticAgent([custom_tool])
```

## Future Enhancements

- Integration with LLMs (OpenAI, Anthropic, etc.)
- More sophisticated reasoning and planning
- Multi-agent collaboration
- Persistent memory storage
- Web interface

## Reference

Based on concepts from: https://www.youtube.com/watch?v=XdbIv7AE3VA

## License

MIT License - Feel free to use and modify!
