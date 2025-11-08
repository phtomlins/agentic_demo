from ollama import generate
# Regular response
response = generate('gemma3', 'Why is the sky blue?')
print(response['response'])

# Streaming response
print("Streaming response:")
for chunk in generate('gemma3', 'Why is the sky blue?', stream=True):
    print(chunk['response'], end='', flush=True)
print()  # New line at the end

