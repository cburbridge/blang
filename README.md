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

for 10 q as n, i {
}


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
