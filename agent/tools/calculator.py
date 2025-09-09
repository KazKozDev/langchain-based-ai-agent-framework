"""
Calculator tool for mathematical computations.
"""

import re
import ast
import operator
from typing import Union
from langchain_core.tools import Tool, StructuredTool
from pydantic import BaseModel, Field


class CalculatorInput(BaseModel):
    """Input schema for calculator tool."""
    expression: str = Field(description="Mathematical expression to calculate")


class CalculatorTool:
    """Calculator tool for safe mathematical operations."""
    
    OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }
    
    def __init__(self):
        """Initialize calculator tool."""
        self.name = "calculator"
        self.description = (
            "Performs mathematical computations. "
            "Supports basic operations: +, -, *, /, **, %, parentheses. "
            "Examples: '2+2', '(5*3)**2', 'sqrt(16)'. "
            "Input must be a mathematical expression."
        )
    
    def _safe_eval(self, expression: str) -> Union[int, float]:
        """
        Safely evaluate mathematical expressions.
        
        Args:
            expression: Mathematical expression as string
            
        Returns:
            Result of calculation
            
        Raises:
            ValueError: If expression is invalid or unsafe
        """
        # Remove whitespace
        expression = expression.replace(" ", "")
        
        # Handle common math functions
        expression = self._replace_math_functions(expression)
        
        try:
            # Parse the expression into AST
            node = ast.parse(expression, mode='eval')
            
            # Evaluate the expression safely
            result = self._eval_node(node.body)
            
            return result
            
        except Exception as e:
            raise ValueError(f"Error in expression '{expression}': {str(e)}")
    
    def _replace_math_functions(self, expression: str) -> str:
        """Replace math functions with their implementations."""
        # Handle sqrt function
        expression = re.sub(r'sqrt\(([^)]+)\)', r'(\1)**0.5', expression)
        
        # Handle abs function  
        expression = re.sub(r'abs\(([^)]+)\)', r'abs(\1)', expression)
        
        return expression
    
    def _eval_node(self, node):
        """Recursively evaluate AST nodes."""
        if isinstance(node, ast.Constant):  # Python 3.8+
            return node.value
        elif isinstance(node, ast.Num):  # Python < 3.8
            return node.n
        elif isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            op = self.OPERATORS[type(node.op)]
            return op(left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand)
            op = self.OPERATORS[type(node.op)]
            return op(operand)
        elif isinstance(node, ast.Call) and node.func.id == 'abs':
            arg = self._eval_node(node.args[0])
            return abs(arg)
        else:
            raise ValueError(f"Unsupported operation: {type(node).__name__}")
    
    def calculate(self, expression: str) -> str:
        """
        Perform calculation and return result as string.
        
        Args:
            expression: Mathematical expression
            
        Returns:
            Calculation result or error message
        """
        try:
            if not expression or not isinstance(expression, str):
                return "Error: Empty expression or wrong data type"
            
            result = self._safe_eval(expression)
            
            # Format result nicely
            if isinstance(result, float) and result.is_integer():
                result = int(result)
            
            return f"Result: {expression} = {result}"
            
        except ValueError as e:
            return f"Calculation error: {str(e)}"
        except ZeroDivisionError:
            return "Error: Division by zero"
        except Exception as e:
            return f"Unexpected error: {str(e)}"
    
    def get_tool(self) -> StructuredTool:
        """Get LangChain StructuredTool instance."""
        return StructuredTool.from_function(
            func=self.calculate,
            name=self.name,
            description=self.description,
            args_schema=CalculatorInput
        )