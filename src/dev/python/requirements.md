General requirements for Python code
- Strict compliance with PEP8.
- The length of the string is 79 characters.
- The imports are properly sorted, there are no unused imports.
- The margins are 4 spaces.
- Hyphenation with correct indentation.
- Backslashes are not used for transfers.
- Consistency (the same quotes, the same methods of solving the same problems, and so on).
- Lack of commented code and standard comments (# Create your views here. etc.).
- Comments on functions are formatted as Docstrings, in accordance with the Docstring
Conventions: Begin with a capital letter, end with a period, and contain a description of what the function does.
- The comments to the code are concise and informative.
- Long pieces of code are logically separated by blank lines like paragraphs in a text.
- There are no unnecessary operations.
- There are no extra else where they are not needed (if a return/raise occurs in the if); Guard Block is used
- There are no unnecessary files in the repository: no pycache , .vscode and other things.
- The executable code in .py files must be closed with the if name == ‘main’ construction.
- For immutable sequences of data, tuples rather than lists are preferable.
- In f-strings, only variable substitution is used and there are no logical or arithmetic operations, function calls, or similar dynamics.
- Variables are named according to their meaning, in English, there are no single-letter
names and transliteration. The variable name should not contain its type. If necessary, type annotations are used.