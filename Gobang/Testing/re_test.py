import re

string = '-11111]'[: -1]
print(string)
result = re.findall('[^-]?11111', string)
print(result)