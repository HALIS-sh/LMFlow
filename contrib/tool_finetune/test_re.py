import re
input_str = """{
 "name": "get_current_weather",
 "arguments": {
  "location": "Dallas, TX"
 }
}"""
# Regex pattern to extract 'name' and 'arguments'
pattern = r'"name":\s*"([^"]+)",\s*"arguments":\s*\{(.*?)\}'
input_str = input_str.replace('\n', '')
print("input_str:", input_str)
match = re.search(pattern, input_str)
# # Use re.DOTALL flag to allow '.' to match newlines
# match = re.search(pattern, input_str, re.DOTALL)
print("match:", match)
if match:
    name = match.group(1)
    arguments = match.group(2)
    print("Name:", name)
    print("Arguments:", arguments)
else:
    print("No match found.")