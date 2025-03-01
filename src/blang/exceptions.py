class UnexpectedCharacterError(Exception):
    def __init__(self, c, line, col):
        self.c = c
        self.line = line
        self.col = col
