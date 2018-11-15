def _check_substr(string, *substrings):
    for substring in substrings:
        if substring in string:
            return True
    return False


line = '1111-101110'
exist = lambda *substrings: _check_substr(line, *substrings)
print(exist('1111-1', '01110'))