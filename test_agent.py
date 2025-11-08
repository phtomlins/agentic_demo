"""
Simple tests for the Agentic AI Demo
Run with: python test_agent.py
"""

from agent import AgenticAgent, Tool, calculator, weather_lookup, text_analyzer


def test_calculator():
    """Test calculator tool."""
    print("Testing calculator...")
    assert calculator("add", 5, 3) == 8
    assert calculator("subtract", 10, 4) == 6
    assert calculator("multiply", 6, 7) == 42
    assert calculator("divide", 20, 4) == 5
    assert calculator("divide", 10, 0) == "Error: Division by zero"
    assert "Error" in calculator("unknown", 1, 2)
    print("✓ Calculator tests passed")


def test_weather():
    """Test weather lookup tool."""
    print("Testing weather lookup...")
    result = weather_lookup("Tokyo")
    assert "Tokyo" in result
    assert "Rainy" in result
    
    result = weather_lookup("Unknown City")
    assert "not available" in result
    print("✓ Weather lookup tests passed")


def test_text_analyzer():
    """Test text analyzer tool."""
    print("Testing text analyzer...")
    result = text_analyzer("Hello world!")
    assert result["character_count"] == 12
    assert result["word_count"] == 2
    assert result["sentence_count"] == 1
    print("✓ Text analyzer tests passed")


def test_agent():
    """Test the agent."""
    print("Testing agent...")
    
    # Create agent with tools
    tools = [
        Tool("calculator", "Math operations", calculator),
        Tool("weather", "Weather info", weather_lookup),
    ]
    agent = AgenticAgent(tools)
    
    # Test tool listing
    tool_list = agent.list_tools()
    assert "calculator" in tool_list
    assert "weather" in tool_list
    
    # Test tool usage
    result = agent.use_tool("calculator", "add", 10, 20)
    assert result == 30
    
    result = agent.use_tool("weather", "London")
    assert "London" in result
    
    # Test error handling
    result = agent.use_tool("nonexistent", "arg")
    assert "Error" in result
    
    # Test memory
    assert len(agent.get_memory()) == 2  # 2 successful tools used
    
    agent.clear_memory()
    assert len(agent.get_memory()) == 0
    
    print("✓ Agent tests passed")


def test_tool_class():
    """Test the Tool class."""
    print("Testing Tool class...")
    
    def sample_func(x):
        return x * 2
    
    tool = Tool("test", "Test tool", sample_func)
    assert tool.name == "test"
    assert tool.description == "Test tool"
    assert tool.execute(5) == 10
    
    print("✓ Tool class tests passed")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running Agentic AI Demo Tests")
    print("=" * 60)
    print()
    
    try:
        test_calculator()
        test_weather()
        test_text_analyzer()
        test_tool_class()
        test_agent()
        
        print()
        print("=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        return True
    except AssertionError as e:
        print()
        print("=" * 60)
        print(f"TEST FAILED ✗: {e}")
        print("=" * 60)
        return False
    except Exception as e:
        print()
        print("=" * 60)
        print(f"ERROR ✗: {e}")
        print("=" * 60)
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
