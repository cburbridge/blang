# todo

## Bugs
- fix:  deref an array item >a[0]<   - >a< + n workaround
- fix:  checking function calls have sufficient parameters
- fix:  deref into self register needs to zero upper bits
- fix:  comment lines are throwing the line numbers

## Feat

- maybe cache builds
- double break or for...else
- type safe overhaul, unify checks
- better parse errors
- add debug tracing with .loc and .file directives
- call func with string literal direct

## Clean
- no need for global strings to go into ro data too

## Core
- floats
- or & and operation
- structs
- bit operations

----------

# Done
DONE   calling functions with floats
DONE :  check stack alignment - its going to be bad
DONE  fix:  intialise with e.g u8|99| - not a const when not a stack var
DONE fix:  assign var to a var
DONE  get printf working
DONE string escapes
DONE tidy: move terminator out of node parses into own parser
DONE  fix calling fuction with wrong type / not ref that should be paramteter
DONE if(1) and if(0) - literal not in reg
DONE xor rax before func call brok test_divide
DONE consider possible of removing {} for whitespace
DONE  not condition
DONE fix bug returning a return
DONE negatives
DONE continue
DONE while
DONE break 
DONE for loop var should promote scope
DONE fix range loops allow varible size
DONE fix squelching of derefs and liteals
DONE multiplication
DONE logicals
DONE if
DONE characters
DONE strings
DONE syscalls
DONE for loops
DONE for loops on non bytes - fix increments
DONE squelching

# Grammar

Statement       -> Assignment | FuncDef | FuncCall |
                   Declaration | ReturnStatement |
                   IfStatement | WhileLoop | ForLoop
Block           -> '{' Statement* '}'
Assignment      -> IDENTIFIER '=' Expr
Declaration     -> TypedIdentifier ('=' Expr)?
ReturnStatement -> 'return' Expr

Expr            -> LogicOr   # skip to Additive

LogicOr         -> LogicAnd ('or' LogicAnd)*      
LogicAnd        -> Equality ('and' Equality)*      
Equality        -> Relational (('==' | '!=') Relational)*  
Relational      -> Additive (('>' | '<' | '>=' | '<=') Additive)*

Additive        -> Term (('+' | '-') Term)*    
Term            -> Factor (('*' | '/') Factor)*
Factor          -> NUMBER | IDENTIFIER | FuncCall | '(' Expr ')' 

BaseType        -> 'u8' | 'u16' | 'u32' | 'u64' | 'u128' | 'i8' | 'i16' | 'i32' | 'i64' | 'i128' | 'f32' | 'f64'
Type            -> BaseType ('ref'*)
Identifier      -> IDENTIFIER
TypedIdentifier -> IDENTIFIER ':' Type

FuncDef         -> 'def' IDENTIFIER '(' TypedIdentifier (',' TypedIdentifier)* ')':' Type '{' Statement* '}'
FuncCall        -> IDENTIFIER '(' (Expr (',' Expr)*)? ')'

IfStatement     -> 'if' '(' Expr ')' Block ('else' Block)?
WhileLoop       -> 'while' '(' Expr ')' Block
ForLoop         -> 'for' '(' Assignment? ';' Expr? ';' Assignment? ')' Block


q:u32[100]

for q as n, i {
}

for 1..100 step 5 as i:u8 {
}

for 1..100 : 5 as i:u8 {
}

u8|a|



a: u32
b: u64

def add(p:u32, q:u32):u32 {
  v: u32
  v = 100
  return p+q+v
}

def main() :u8 {
   add(9,10)
   return 0
}
