#!/usr/bin/env python3
"""
Interactive example showing how to use the Agentic AI Demo
"""

from agent import AgenticAgent, Tool, calculator, weather_lookup, text_analyzer


def main():
    """Run an interactive demonstration."""
    
    # Create tools
    tools = [
        Tool("calculator", "Perform arithmetic operations (add, subtract, multiply, divide)", calculator),
        Tool("weather", "Get weather information for a city", weather_lookup),
        Tool("text_analyzer", "Analyze text and get statistics", text_analyzer)
    ]
    
    # Create agent
    agent = AgenticAgent(tools)
    
    print("Welcome to the Agentic AI Interactive Demo!")
    print("-" * 60)
    print()
    print(agent.list_tools())
    print()
    print("Examples of what you can do:")
    print("1. agent.use_tool('calculator', 'multiply', 6, 7)")
    print("2. agent.use_tool('weather', 'London')")
    print("3. agent.use_tool('text_analyzer', 'Your text here')")
    print()
    print("-" * 60)
    print()
    
    # Scenario: Planning a trip
    print("SCENARIO: Planning a trip calculation")
    print("=" * 60)
    print()
    
    # Calculate total cost
    print("Step 1: Calculate flight cost for 2 people at $450 each")
    flight_cost = agent.use_tool("calculator", "multiply", 450, 2)
    print(f"Flight cost: ${flight_cost}")
    print()
    
    # Calculate hotel cost
    print("Step 2: Calculate hotel cost for 5 nights at $120 per night")
    hotel_cost = agent.use_tool("calculator", "multiply", 120, 5)
    print(f"Hotel cost: ${hotel_cost}")
    print()
    
    # Calculate total
    print("Step 3: Calculate total trip cost")
    total_cost = agent.use_tool("calculator", "add", flight_cost, hotel_cost)
    print(f"Total trip cost: ${total_cost}")
    print()
    
    # Check weather at destination
    print("Step 4: Check weather at destination")
    weather = agent.use_tool("weather", "Paris")
    print(weather)
    print()
    
    print("=" * 60)
    print()
    
    # Show agent's memory
    print("Agent's Memory of Actions:")
    print("-" * 60)
    for i, mem in enumerate(agent.get_memory(), 1):
        print(f"{i}. Used '{mem['tool']}' with args {mem.get('args', ())} -> Result: {mem.get('result', 'N/A')}")
    print()
    
    # Demonstrate error handling
    print("=" * 60)
    print("DEMONSTRATION: Error Handling")
    print("=" * 60)
    print()
    
    print("Attempting division by zero:")
    result = agent.use_tool("calculator", "divide", 10, 0)
    print(f"Result: {result}")
    print()
    
    print("Attempting to use non-existent tool:")
    result = agent.use_tool("nonexistent_tool", "some_arg")
    print(f"Result: {result}")
    print()
    
    print("Demo completed successfully!")


if __name__ == "__main__":
    main()
