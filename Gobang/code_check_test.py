#!/usr/bin/env python3
import sys
from code_check import CodeCheck


def main():
    code_checker = CodeCheck("C:\\Users\\10578\\PycharmProjects\\AICourse\\Gobang\\GobangAINext.py", 15)
    if not code_checker.check_code():
        print(code_checker.errormsg)
    else:
        print('pass')


if __name__ == '__main__':
    main()
