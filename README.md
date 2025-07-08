# Mathematical Expression Parser and Evaluator

This project is a Python-based mathematical expression parser and evaluator that supports arithmetic operations, mathematical functions, and variables. It can convert infix expressions to postfix notation, evaluate them, and plot functions with a single variable (`x`) over a specified interval. It also includes a root-finding feature using the bisection method.

## Features
- **Arithmetic Operations**: Supports `+`, `-`, `*`, `/`, `^` (exponentiation), `div` (integer division), and `mod` (modulus).
- **Mathematical Functions**: Includes `sin`, `cos`, `tan`, `log`, `sqrt`, `exp`, `arcsin`, `arccos`, `arctan`, `cotan`, `arccotan`, and `sqr`.
- **Constants**: Supports `pi` and `e`.
- **Variables**: Allows user-defined variables with interactive input for values.
- **Comments**: Supports block comments `{...}` and line comments `//`.
- **Graphing**: Plots functions with a single variable (`x`) using Matplotlib.
- **Root Finding**: Finds roots in a specified interval using the bisection method.

## Requirements
- Python 3.6+
- Required libraries:
  - `numpy`
  - `matplotlib`

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```
2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Run the script:
   ```bash
   python calculator.py
   ```
2. At the `spi>` prompt, enter a mathematical expression, such as:
   ```
   sin(x * pi / 180) + 2^3
   ```
3. If the expression contains variables (other than `x`, `pi`, or `e`), you will be prompted to enter their values.
4. If the expression contains `x`, you will be prompted to enter an interval `[a, b]` for graphing and optionally find a root in that interval.
5. The program outputs the postfix notation, the evaluated result, and (if applicable) a plot of the function.

### Example
```
spi> sin(x * pi / 180) + 2^3
Postfix notation: x pi * 180 / sin 8 +
Enter value for variable 'x': 90
Result: 9.0
Enter interval [a, b] for graphing (e.g., -10 10): -180 180
Would you like to find a root in this interval? (yes/no): no
[Graph is displayed]
```

## Project Structure
- `calculator.py`: Main script containing the lexer, parser, evaluator, and plotting functionality.
- `requirements.txt`: Lists the required Python libraries.
- `.gitignore`: Specifies files and directories to ignore in version control.

## Contributing
Contributions are welcome! Please fork the repository, make your changes, and submit a pull request.

## License
This project is licensed under the MIT License.
