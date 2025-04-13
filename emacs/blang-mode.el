(defvar blang-mode-hook nil)

(defvar blang-mode-map
  (let ((map (make-keymap)))
    ;; Add custom bindings here
    map)
  "Keymap for BLang major mode")

(add-to-list 'auto-mode-alist '("\\.bl\\'" . blang-mode))

(defconst blang-font-lock-keywords
  (let* (
         (keywords '("if" "else" "while" "for" "return" "break" "external" "export " "continue" "as" "def"))
         (types '("u8" "u16" "u32" "u64" "f64"))
         (constants '("true" "false" "null" "NULL"))

         (keywords-regexp (regexp-opt keywords 'words))
         (types-regexp (regexp-opt types 'words))
         (constants-regexp (regexp-opt constants 'words))
         )
    `(
      (,keywords-regexp . font-lock-keyword-face)
      (,types-regexp . font-lock-type-face)
      (,constants-regexp . font-lock-constant-face)
      ;; Function names: def foo(...) { ... }
      ("def[ \t]+\\([a-zA-Z_][a-zA-Z0-9_]*\\)" 1 font-lock-function-name-face)
      ;; Pointer declaration: d: <u8>
      ("\\([a-zA-Z_][a-zA-Z0-9_]*\\): *<\\([a-zA-Z0-9_]+\\)>" 
       (1 font-lock-variable-name-face) (2 font-lock-type-face))
      ;; Dereference: >d<
      (">\\([a-zA-Z_][a-zA-Z0-9_]*\\)<" 1 font-lock-variable-name-face)
      ;; String literals
      ("\"[^\"]*\"" . font-lock-string-face)
      )))
(defun blang-indent-line ()
  "Indent current line for BLang."
  (interactive)
  (let ((cur-indent 0)
        (offset 4)
        (pos (- (point-max) (point))))
    (save-excursion
      (beginning-of-line)
      (cond
       ((bobp)
        (setq cur-indent 0))
       ((looking-at "^[ \t]*\\}")
        (forward-line -1)
        (while (and (not (bobp)) (looking-at "^[ \t]*$"))
          (forward-line -1))
        (setq cur-indent (- (current-indentation) offset)))
       (t
        (forward-line -1)
        (while (and (not (bobp)) (looking-at "^[ \t]*$"))
          (forward-line -1))
        (setq cur-indent (current-indentation))
        (when (looking-at ".*{[ \t]*$")
          (setq cur-indent (+ cur-indent offset))))))
    (indent-line-to (max cur-indent 0))))

(define-derived-mode blang-mode prog-mode "BLang"
  "Major mode for editing BLang source code."
  (setq font-lock-defaults '((blang-font-lock-keywords)))
  (setq comment-start "#")
  (setq comment-end "")
  (setq-local indent-line-function 'blang-indent-line))
