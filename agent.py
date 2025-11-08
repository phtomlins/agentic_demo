"""
Simple Agentic AI Demo
A demonstration of an AI agent that can use tools to accomplish tasks.
"""

import json
from typing import List, Dict, Callable, Any


class Tool:
    """Represents a tool that the agent can use."""
    
    def __init__(self, name: str, description: str, func: Callable):
        self.name = name
        self.description = description
        self.func = func
    
    def execute(self, *args, **kwargs) -> Any:
        """Execute the tool function."""
        return self.func(*args, **kwargs)


class AgenticAgent:
    """
    A simple agentic AI agent that can reason about tasks and use tools.
    This is a simplified demonstration without actual LLM integration.
    """
    
    def __init__(self, tools: List[Tool]):
        self.tools = {tool.name: tool for tool in tools}
        self.memory = []
    
    def list_tools(self) -> str:
        """List available tools."""
        tool_list = "\n".join([f"- {name}: {tool.description}" 
                               for name, tool in self.tools.items()])
        return f"Available tools:\n{tool_list}"
    
    def use_tool(self, tool_name: str, *args, **kwargs) -> Any:
        """Use a specific tool by name."""
        if tool_name not in self.tools:
            return f"Error: Tool '{tool_name}' not found. Available tools: {list(self.tools.keys())}"
        
        tool = self.tools[tool_name]
        try:
            result = tool.execute(*args, **kwargs)
            self.memory.append({
                "action": "tool_use",
                "tool": tool_name,
                "args": args,
                "kwargs": kwargs,
                "result": result
            })
            return result
        except Exception as e:
            error_msg = f"Error executing {tool_name}: {str(e)}"
            self.memory.append({
                "action": "tool_use",
                "tool": tool_name,
                "error": error_msg
            })
            return error_msg
    
    def get_memory(self) -> List[Dict]:
        """Get the agent's memory of past actions."""
        return self.memory
    
    def clear_memory(self):
        """Clear the agent's memory."""
        self.memory = []


# Example tool functions
def calculator(operation: str, a: float, b: float) -> float:
    """
    Perform basic arithmetic operations.
    
    Args:
        operation: One of 'add', 'subtract', 'multiply', 'divide'
        a: First number
        b: Second number
    
    Returns:
        Result of the operation
    """
    operations = {
        'add': lambda x, y: x + y,
        'subtract': lambda x, y: x - y,
        'multiply': lambda x, y: x * y,
        'divide': lambda x, y: x / y if y != 0 else "Error: Division by zero"
    }
    
    if operation not in operations:
        return f"Error: Unknown operation '{operation}'"
    
    return operations[operation](a, b)


def weather_lookup(city: str) -> str:
    """
    Mock weather lookup function.
    
    Args:
        city: Name of the city
    
    Returns:
        Weather information (mocked)
    """
    # Mock weather data
    weather_data = {
        "new york": "Sunny, 72°F",
        "london": "Cloudy, 60°F",
        "tokyo": "Rainy, 68°F",
        "paris": "Partly cloudy, 65°F",
        "sydney": "Clear, 75°F"
    }
    
    city_lower = city.lower()
    if city_lower in weather_data:
        return f"Weather in {city}: {weather_data[city_lower]}"
    else:
        return f"Weather data not available for {city}"


def text_analyzer(text: str) -> Dict[str, Any]:
    """
    Analyze text and return basic statistics.
    
    Args:
        text: Text to analyze
    
    Returns:
        Dictionary with text statistics
    """
    words = text.split()
    return {
        "character_count": len(text),
        "word_count": len(words),
        "sentence_count": text.count('.') + text.count('!') + text.count('?'),
        "average_word_length": sum(len(word) for word in words) / len(words) if words else 0
    }


if __name__ == "__main__":
    # Create tools
    tools = [
        Tool("calculator", "Perform arithmetic operations (add, subtract, multiply, divide)", calculator),
        Tool("weather", "Get weather information for a city", weather_lookup),
        Tool("text_analyzer", "Analyze text and get statistics", text_analyzer)
    ]
    
    # Create agent
    agent = AgenticAgent(tools)
    
    # Demonstrate agent capabilities
    print("=" * 60)
    print("AGENTIC AI DEMO")
    print("=" * 60)
    print()
    
    print(agent.list_tools())
    print()
    
    # Example 1: Use calculator
    print("Example 1: Calculate 15 + 27")
    result = agent.use_tool("calculator", "add", 15, 27)
    print(f"Result: {result}")
    print()
    
    # Example 2: Check weather
    print("Example 2: Check weather in Tokyo")
    result = agent.use_tool("weather", "Tokyo")
    print(f"Result: {result}")
    print()
    
    # Example 3: Analyze text
    print("Example 3: Analyze sample text")
    sample_text = "The quick brown fox jumps over the lazy dog. This is a demo."
    result = agent.use_tool("text_analyzer", sample_text)
    print(f"Result: {json.dumps(result, indent=2)}")
    print()
    
    # Show memory
    print("Agent Memory:")
    for i, mem in enumerate(agent.get_memory(), 1):
        print(f"{i}. {mem}")
    print()
    
    print("=" * 60)
    print("Demo completed!")
    print("=" * 60)
