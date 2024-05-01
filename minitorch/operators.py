"""
Collection of the core mathematical operators used throughout the code base.
"""

import math
from typing import Callable, Iterable

# ## Task 0.1
#
# Implementation of a prelude of elementary functions.


def mul(x: float, y: float) -> float:
    """
    Multiply two numbers.

    This function takes two floating-point numbers and returns their product.
    It demonstrates a basic arithmetic operation implemented as a function.

    Args:
        x (float): The first number.
        y (float): The second number.

    Returns:
        float: The product of x and y.
    """
    return x * y


def id(x: float) -> float:
    """
    Return the input without any modification.

    This identity function demonstrates the simplest form of a function
    that returns exactly what it receives. It's often used for demonstrations
    and as a placeholder in computational graphs or during testing.

    Args:
        x (float): A numeric value.

    Returns:
        float: The same numeric value passed as input.
    """
    return x


def add(x: float, y: float) -> float:
    """
    Add two numbers.

    This function takes two floating-point numbers and returns their sum. It
    demonstrates a basic arithmetic operation.

    Args:
        x (float): The first number to add.
        y (float): The second number to add.

    Returns:
        float: The sum of x and y.
    """
    return x + y


def neg(x: float) -> float:
    """
    Negate the input value.

    This function returns the negative of the given floating-point number. It is
    a straightforward arithmetic operation that changes the sign of the number.

    Args:
        x (float): The number to negate.

    Returns:
        float: The negated value of x.
    """
    return -x


def lt(x: float, y: float) -> float:
    """
    Determine if one number is less than another.

    This function compares two floating-point numbers and returns 1.0 if the first
    number is less than the second number, and 0.0 otherwise. It's used to perform
    element-wise comparisons in vectorized operations.

    Args:
        x (float): The first number to compare.
        y (float): The second number to compare against.

    Returns:
        float: 1.0 if x is less than y, otherwise 0.0.
    """
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """
    Determine if two numbers are equal.

    This function compares two floating-point numbers and returns 1.0 if they are equal,
    and 0.0 otherwise. It is useful for element-wise equality checks in numerical computations.

    Args:
        x (float): The first number to compare.
        y (float): The second number to compare.

    Returns:
        float: 1.0 if x is equal to y, otherwise 0.0.
    """
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """
    Return the maximum of two numbers.

    This function compares two floating-point numbers and returns the larger of the two.
    It is a fundamental operation used in various numerical and data processing tasks
    to determine the greater value between two operands.

    Args:
        x (float): The first number to compare.
        y (float): The second number to compare.

    Returns:
        float: The greater number between x and y.
    """
    return x if x > y else y


def is_close(x: float, y: float) -> bool:
    """
    Determine if two numbers are approximately equal to each other within a small tolerance.

    This function evaluates the absolute difference between two floating-point numbers
    and checks if it is less than a specified tolerance level (0.01). It is typically used
    in situations where floating-point precision might lead to small discrepancies in
    calculations, making exact comparison impractical.

    Args:
        x (float): The first number.
        y (float): The second number.

    Returns:
        bool: True if the difference between x and y is less than 0.01, else False.
    """
    return abs(x - y) < 0.01


def sigmoid(x: float) -> float:
    r"""
    $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$

    (See https://en.wikipedia.org/wiki/Sigmoid_function )

    Calculate as

    $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$

    for stability.
    """
    # TODO: Implement for Task 0.1.
    raise NotImplementedError("Need to implement for Task 0.1")


def relu(x: float) -> float:
    """
    Apply the rectified linear unit function.

    This function returns x if x is greater than zero, otherwise it returns zero.
    It is commonly used as an activation function in neural networks.

    Args:
        x (float): The input value.

    Returns:
        float: The output of the ReLU function.

    See Also:
        https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
    """
    return x if x > 0 else 0


EPS = 1e-6


def log(x: float) -> float:
    "$f(x) = log(x)$"
    return math.log(x + EPS)


def exp(x: float) -> float:
    "$f(x) = e^{x}$"
    return math.exp(x)


def log_back(x: float, d: float) -> float:
    r"If $f = log$ as above, compute $d \times f'(x)$"
    # TODO: Implement for Task 0.1.
    raise NotImplementedError("Need to implement for Task 0.1")


def inv(x: float) -> float:
    """
    Calculate the reciprocal of a number.

    This function returns the reciprocal of the given floating-point number. It handles
    the basic arithmetic operation of finding the inverse of a number, which is useful
    in various mathematical computations.

    Args:
        x (float): The number to find the reciprocal of.

    Returns:
        float: The reciprocal of x.

    Raises:
        ValueError: If x is zero, as division by zero is undefined.
    """
    if x == 0:
        raise ValueError("Division by zero is undefined.")
    return 1 / x


def inv_back(x: float, d: float) -> float:
    r"If $f(x) = 1/x$ compute $d \times f'(x)$"
    # TODO: Implement for Task 0.1.
    raise NotImplementedError("Need to implement for Task 0.1")


def relu_back(x: float, d: float) -> float:
    """
    Compute the gradient of the ReLU function for backpropagation.

    This function calculates the derivative of the ReLU activation function
    applied to an input x, scaled by the gradient of the loss with respect to
    the output of the ReLU function (d). This is essential for the backpropagation
    process in neural networks to update the weights.

    Args:
        x (float): The input value to the ReLU function.
        d (float): The upstream gradient.

    Returns:
        float: The gradient of ReLU with respect to input x, scaled by d.
    """
    return d * (1 if x > 0 else 0)


# ## Task 0.3

# Small practice library of elementary higher-order functions.


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """
    Higher-order map.

    See https://en.wikipedia.org/wiki/Map_(higher-order_function)

    Args:
        fn: Function from one value to one value.

    Returns:
        A function that takes a list, applies `fn` to each element, and returns a
         new list
    """
    # TODO: Implement for Task 0.3.
    raise NotImplementedError("Need to implement for Task 0.3")


def negList(ls: Iterable[float]) -> Iterable[float]:
    "Use `map` and `neg` to negate each element in `ls`"
    # TODO: Implement for Task 0.3.
    raise NotImplementedError("Need to implement for Task 0.3")


def zipWith(
    fn: Callable[[float, float], float]
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """
    Higher-order zipwith (or map2).

    See https://en.wikipedia.org/wiki/Map_(higher-order_function)

    Args:
        fn: combine two values

    Returns:
        Function that takes two equally sized lists `ls1` and `ls2`, produce a new list by
         applying fn(x, y) on each pair of elements.

    """
    # TODO: Implement for Task 0.3.
    raise NotImplementedError("Need to implement for Task 0.3")


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    "Add the elements of `ls1` and `ls2` using `zipWith` and `add`"
    # TODO: Implement for Task 0.3.
    raise NotImplementedError("Need to implement for Task 0.3")


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    r"""
    Higher-order reduce.

    Args:
        fn: combine two values
        start: start value $x_0$

    Returns:
        Function that takes a list `ls` of elements
         $x_1 \ldots x_n$ and computes the reduction :math:`fn(x_3, fn(x_2,
         fn(x_1, x_0)))`
    """
    # TODO: Implement for Task 0.3.
    raise NotImplementedError("Need to implement for Task 0.3")


def sum(ls: Iterable[float]) -> float:
    "Sum up a list using `reduce` and `add`."
    # TODO: Implement for Task 0.3.
    raise NotImplementedError("Need to implement for Task 0.3")


def prod(ls: Iterable[float]) -> float:
    "Product of a list using `reduce` and `mul`."
    # TODO: Implement for Task 0.3.
    raise NotImplementedError("Need to implement for Task 0.3")
