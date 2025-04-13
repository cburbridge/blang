section .text
    global print_string

; def print_string(str : <u8>, len: u32)
; rdi = pointer to string
; rsi = length of string

print_string:
    mov     rax, 1        ; syscall number for write
    mov     rdx, rsi      ; rdx = length
    mov     rsi, rdi      ; rsi = pointer to string
    mov     rdi, 1        ; rdi = stdout (fd = 1)
    syscall
    ret
