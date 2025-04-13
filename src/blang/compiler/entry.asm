global _start
extern main
; --- ENTRY POINT ---
_start:
  call main
  mov edi, eax
  mov eax, 60
  syscall
