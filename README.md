[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

# BLang - A Minimal Programming Language

This is a toy programming language with its own compiler backend that emits x86_64 assembly. It supports basic types, arrays, pointers, functions, conditionals, loops, etc. The compiler is written in Python. 

## Why?

Just for fun. I wanted to learn how compilers work at a deeper level, and this was a great way to explore parsing, code generation, memory layout, calling conventions, etc.

## Is it useful?

Probably not. It's not fast, it's not safe, it's not feature-complete, and it doesn't have a standard library. But the code is (mostly) readable, and it covers a lot of core ideas behind compiling a high-level language down to raw machine code.

## Running

Install NASM. 

To compile a .bl source file to a native executable:

blangc your_program.bl -o your_program

To run tests:

pytest tests/

## Current Features

- Integers (signed/unsigned), doubles
- Arrays and pointers
- Functions (multiple arguments, stack handling)
- Conditionals (if, else)
- Loops (while, for)
- Bitwise operations (&, |, ^)
- String literals and basic string handling
- Assembly output using NASM
- Simple CLI compiler
- CI build and test pipeline

## Missing / Incomplete / Won't Do

- No type inference or advanced type checking
- No heap allocation or garbage collection
- No standard library
- Limited error reporting
- No optimization passes
- No inline assembly or macros (yet)
- No structs

## License

MIT

## Example programs
See `examples/`