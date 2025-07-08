import math
import matplotlib.pyplot as plt
import numpy as np


# Token types
INTEGER, FLOAT, PLUS, MINUS, MUL, DIV, INTDIV, MOD, EXP, LPAREN, RPAREN, FUNC, CONSTANT, ID, EOF = (
    'INTEGER', 'FLOAT', 'PLUS', 'MINUS', 'MUL', 'DIV', 'INTDIV', 'MOD', 'EXP',
    '(', ')', 'FUNC', 'CONSTANT', 'ID', 'EOF'
)


class Token:
    def __init__(self, type, value, line=None):
        self.type = type
        self.value = value
        self.line = line  # Line number where the token is located

    def __str__(self):
        return f"Token({self.type}, {repr(self.value)}, line={self.line})"

    def __repr__(self):
        return self.__str__()


class Lexer:
    FUNCTIONS = {'sin', 'cos', 'tan', 'log', 'sqrt', 'exp', 'arcsin', 'arccos', 'arctan', 'cotan', 'arccotan', 'sqr'}

    def __init__(self, text):
        self.text = text
        self.pos = 0
        self.line = 1
        self.current_char = text[self.pos] if text else None

    def error(self):
        raise Exception(f"Invalid character at line {self.line}")

    def advance(self):
        if self.current_char == '\n':  # Increment line number on newline
            self.line += 1
        self.pos += 1
        self.current_char = self.text[self.pos] if self.pos < len(self.text) else None

    def peek(self):
        peek_pos = self.pos + 1
        return self.text[peek_pos] if peek_pos < len(self.text) else None

    def skip_whitespace(self):
        while self.current_char and self.current_char.isspace():
            self.advance()

    def skip_comment(self):
        if self.current_char == '{':  # Block comment
            self.advance()
            while self.current_char and self.current_char != '}':
                self.advance()
            if self.current_char == '}':
                self.advance()
            else:
                raise Exception(f"Unclosed block comment at line {self.line}")
        elif self.current_char == '/' and self.peek() == '/':  # Line comment
            self.advance()  # Skip the first '/'
            self.advance()  # Skip the second '/'
            while self.current_char and self.current_char != '\n':
                self.advance()

    def number(self):
        """Handle integers, floating-point numbers, and scientific notation."""
        result = ''
        is_float = False
        is_scientific = False

        while self.current_char and (self.current_char.isdigit() or self.current_char in '.Ee'):
            if self.current_char == '.':
                if is_float or is_scientific:  # Invalid number if we already saw '.' or 'E/e'
                    self.error()
                is_float = True
            elif self.current_char in 'Ee':
                if is_scientific:  # Invalid if we already saw 'E/e'
                    self.error()
                is_scientific = True
                result += self.current_char
                self.advance()
                if self.current_char in '+-':  # Handle optional sign in the exponent
                    result += self.current_char
                    self.advance()
                if not self.current_char.isdigit():  # Exponent must be followed by digits
                    self.error()
                continue

            result += self.current_char
            self.advance()

        return Token(FLOAT, float(result), self.line) if (is_float or is_scientific) else Token(INTEGER, int(result),
                                                                                                self.line)

    def identifier(self):
        """Handle identifiers."""
        result = ''
        while self.current_char and (self.current_char.isalnum() or self.current_char == '_'):
            result += self.current_char
            self.advance()

        result = result.lower()

        if result == 'div':  # Recognize 'div' as integer division
            return Token(INTDIV, 'div', self.line)
        elif result == 'mod':  # Recognize 'mod' as modulus
            return Token(MOD, 'mod', self.line)

        # Check if the identifier is a constant or function
        if result == 'pi':
            return Token(CONSTANT, 'pi', self.line)
        elif result == 'e':
            return Token(CONSTANT, 'e', self.line)
        if result in self.FUNCTIONS:
            return Token(FUNC, result, self.line)
        return Token(ID, result, self.line)

    def get_next_token(self):
        while self.current_char:
            if self.current_char.isspace():
                self.skip_whitespace()
                continue
            if self.current_char == '{' or (self.current_char == '/' and self.peek() == '/'):
                self.skip_comment()
                continue
            if self.current_char.isdigit() or (self.current_char == '.' and self.peek() and self.peek().isdigit()):
                return self.number()
            if self.current_char.isalpha() or self.current_char == '_':
                return self.identifier()
            if self.current_char == '+':
                self.advance()
                return Token(PLUS, '+', self.line)
            if self.current_char == '-':
                self.advance()
                return Token(MINUS, '-', self.line)
            if self.current_char == '*':
                self.advance()
                return Token(MUL, '*', self.line)
            if self.current_char == '/':
                self.advance()
                return Token(DIV, '/', self.line)
            if self.current_char == '^':
                self.advance()
                return Token(EXP, '^', self.line)
            if self.current_char == '(':
                self.advance()
                return Token(LPAREN, '(', self.line)
            if self.current_char == ')':
                self.advance()
                return Token(RPAREN, ')', self.line)
            self.error()
        return Token(EOF, None, self.line)


class Parser:
    def __init__(self, lexer):
        self.lexer = lexer
        self.current_token = lexer.get_next_token()

    def error(self):
        raise Exception(f"Invalid syntax near '{self.current_token.value}'")

    def eat(self, token_type):
        if self.current_token.type == token_type:
            self.current_token = self.lexer.get_next_token()
        else:
            self.error()

    def factor(self):
        token = self.current_token
        if token.type in (INTEGER, FLOAT, CONSTANT):
            self.eat(token.type)
            return str(token.value)
        elif token.type == ID:
            self.eat(ID)
            return token.value
        elif token.type == PLUS or token.type == MINUS:
            self.eat(token.type)
            operand = self.factor()
            return f"{operand} neg" if token.type == MINUS else operand
        elif token.type == FUNC:
            func_name = token.value
            self.eat(FUNC)
            if self.current_token.type != LPAREN:
                raise Exception(f"ERROR in line {self.lexer.line}: Missing '(' after function {func_name}")
            self.eat(LPAREN)
            arg = self.expr()
            if self.current_token.type != RPAREN:
                raise Exception(f"ERROR in line {self.lexer.line}: Missing ')' after function {func_name}")
            self.eat(RPAREN)
            return f"{arg} {func_name}"
        elif token.type == LPAREN:
            self.eat(LPAREN)
            node = self.expr()
            if self.current_token.type != RPAREN:
                raise Exception(f"ERROR in line {self.lexer.line}: Unmatched '('")
            self.eat(RPAREN)
            return node
        else:
            raise Exception(f"ERROR in line {self.lexer.line}: Unexpected token '{token.value}'")

    def power(self):
        """Handles the exponentiation operator (^)."""
        left = self.factor()
        while self.current_token.type == EXP:
            token = self.current_token
            self.eat(EXP)
            right = self.power()
            left = f"{left} {right} {token.value}"
        return left

    def term(self):
        """Handles multiplication, division, integer division, and modulus."""
        left = self.power()
        while self.current_token.type in (MUL, DIV, INTDIV, MOD):
            token = self.current_token
            self.eat(token.type)
            right = self.power()
            left = f"{left} {right} {token.value}"

        # Check if the next token is invalid (e.g., an identifier without an operator)
        if self.current_token.type in (INTEGER, FLOAT, CONSTANT, ID):
            raise Exception(f"ERROR in line {self.lexer.line}: Missing operator before '{self.current_token.value}'")

        return left

    def expr(self):
        """Handles addition and subtraction."""
        left = self.term()
        while self.current_token.type in (PLUS, MINUS):
            token = self.current_token
            self.eat(token.type)
            right = self.term()
            left = f"{left} {right} {token.value}"

        # Check for invalid continuation between lines
        if self.current_token.type in (INTEGER, FLOAT, CONSTANT, ID):
            raise Exception(f"ERROR in line {self.lexer.line}: Missing operator before '{self.current_token.value}'")

        return left


    def parse(self):
        return self.expr()


class SymbolTable:
    """Stores variable names and their values."""

    def __init__(self):
        self.table = {}

    def get(self, name):
        return self.table.get(name)

    def set(self, name, value):
        self.table[name] = value

    def prompt_value(self, name):
        """Prompt the user for a variable value if it's not already set."""
        if self.get(name) is None:
            while True:
                try:
                    value = float(input(f"Enter value for variable '{name}': "))
                    self.set(name, value)
                    break
                except ValueError:
                    print("Invalid input. Please enter a numerical value.")


class PostfixEvaluator:
    def __init__(self, postfix, symbol_table):
        self.expression = postfix.split()
        self.symbol_table = symbol_table
        self.stack = []

    def is_float(self, token):
        try:
            float(token)
            return True
        except ValueError:
            return False

    def find_root(self, a, b, tolerance=1e-6, max_iterations=100):
        """Find a root in the interval [a, b] using the bisection method."""
        f_a = self.evaluate_at(a)
        f_b = self.evaluate_at(b)

        if f_a * f_b > 0:
            raise ValueError("The function must have opposite signs at the endpoints.")

        for _ in range(max_iterations):
            c = (a + b) / 2
            f_c = self.evaluate_at(c)

            if abs(f_c) < tolerance or abs(b - a) < tolerance:
                return c

            if f_a * f_c < 0:
                b = c
                f_b = f_c
            else:
                a = c
                f_a = f_c

        raise ValueError("Root not found within the maximum number of iterations.")

    def evaluate_at(self, x):
        """Evaluate the postfix expression at a specific value of x."""
        self.symbol_table.set('x', x)
        return self.evaluate()

    def evaluate(self):
        for token in self.expression:
            if token == 'neg':  # Handle unary negative
                self.stack.append(-self.stack.pop())
            elif token in {'pi', 'e'}:
                if token == 'pi':
                    self.stack.append(math.pi)
                if token == 'e':
                    self.stack.append(math.e)
            # Check for numbers
            elif self.is_float(token):
                self.stack.append(float(token))
            # Check for operators
            elif token in {'+', '-', '*', '/', '^', 'div', 'mod'}:
                b = self.stack.pop()
                a = self.stack.pop()

                if token == 'div' or token == 'mod':  # Validate integer operands
                    if not (a.is_integer() and b.is_integer()):
                        raise TypeError(f"Operands for '{token}' must be integers, got {a} and {b}.")
                    a, b = int(a), int(b)  # Convert to int for these operations

                if token == '+':
                    self.stack.append(a + b)
                elif token == '-':
                    self.stack.append(a - b)
                elif token == '*':
                    self.stack.append(a * b)
                elif token == '/':
                    self.stack.append(a / b)
                elif token == 'div':  # Handle 'div'
                    self.stack.append(a // b)
                elif token == 'mod':  # Handle 'mod'
                    self.stack.append(a % b)
                elif token == '^':
                    self.stack.append(a ** b)
            # Check for functions
            elif token in Lexer.FUNCTIONS:
                arg = self.stack.pop()
                if token == 'sin':
                    self.stack.append(math.sin(arg))
                elif token == 'cos':
                    self.stack.append(math.cos(arg))
                elif token == 'tan':
                    self.stack.append(math.tan(arg))
                elif token == 'log':
                    self.stack.append(math.log(arg))
                elif token == 'sqr':
                    self.stack.append(arg ** 2)
                elif token == 'sqrt':
                    self.stack.append(math.sqrt(arg))
                elif token == 'exp':
                    self.stack.append(math.exp(arg))
                elif token == 'arcsin':
                    self.stack.append(math.asin(arg))
                elif token == 'arccos':
                    self.stack.append(math.acos(arg))
                elif token == 'arctan':
                    self.stack.append(math.atan(arg))
                elif token == 'cotan':
                    self.stack.append(1 / math.tan(arg))
                elif token == 'arccotan':
                    self.stack.append(math.atan(1 / arg))
                else:
                    raise ValueError(f"Unknown function: {token}")
            # Check for variables
            elif (token[0].isalpha() or token[0] == '_') and all(c.isalnum() or c == '_' for c in token):
                value = self.symbol_table.get(token)
                if value is None:
                    raise ValueError(f"Variable '{token}' not found in symbol table.")
                self.stack.append(value)
            else:
                raise ValueError(f"Unknown token: {token}")
        return self.stack.pop()


def plot_function(postfix, symbol_table, interval):
    """
    Plot the graph of a function represented in postfix notation over a given interval.

    :param postfix: The postfix notation of the function.
    :param symbol_table: SymbolTable instance with variable values.
    :param interval: Tuple (a, b) representing the interval [a, b].
    """
    a, b = interval
    x_values = np.linspace(a, b, 1000)  # Generate 1000 points between a and b
    y_values = []

    for x in x_values:
        # Set the value of 'x' in the symbol table for each point
        symbol_table.set('x', x)

        # Evaluate the function at the current 'x'
        evaluator = PostfixEvaluator(postfix, symbol_table)
        try:
            y = evaluator.evaluate()
            y_values.append(y)
        except Exception as e:
            y_values.append(float('nan'))  # Handle evaluation errors gracefully (e.g., log of a negative number)

    # Plot the function
    plt.figure(figsize=(8, 6))
    plt.plot(x_values, y_values, label="f(x)", color="blue")
    plt.axhline(0, color="black", linewidth=0.8, linestyle="--")  # x-axis
    plt.axvline(0, color="black", linewidth=0.8, linestyle="--")  # y-axis
    plt.title("Graph of the Function")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    symbol_table = SymbolTable()

    while True:
        try:
            text = input('spi> ')
        except EOFError:
            break
        if not text:
            continue

        lexer = Lexer(text)
        parser = Parser(lexer)
        try:
            postfix = parser.parse()
            print(f"Postfix notation: {postfix.replace('neg', '-')}")
            # Prompt for variables
            variables = set(
                token for token in postfix.split()
                if (token[0].isalpha() or token[0] == '_') and all(c.isalnum() or c == '_' for c in token)
            )
            for var in variables - Lexer.FUNCTIONS - {'div', 'mod', 'neg', 'pi', 'e'}:
                symbol_table.prompt_value(var)

            evaluator = PostfixEvaluator(postfix, symbol_table)

            # Evaluate the expression
            result = evaluator.evaluate()
            result = 0 if abs(result) < 1e-10 else result
            print(f"Result: {int(result) if result == int(result) else float(result)}")

            # If the expression contains 'x', offer graphing and root finding options
            if 'x' in variables and len(variables) == 1:
                try:
                    interval_input = input("Enter interval [a, b] for graphing (e.g., -10 10): ")
                    a, b = map(float, interval_input.split())

                    # Optionally find a root in the interval
                    find_root_option = input(
                        "Would you like to find a root in this interval? (yes/no): ").strip().lower()
                    if find_root_option == 'yes':
                        try:
                            root = evaluator.find_root(a, b)
                            print(f"Root found at x = {root}")
                        except ValueError as e:
                            print(f"Root finding error: {e}")

                    # Plot the function
                    plot_function(postfix, symbol_table, (a, b))
                except ValueError:
                    print("Invalid interval.")
                except Exception as e:
                    print(e)

        except Exception as e:
            print(e)


if __name__ == '__main__':
    main()
