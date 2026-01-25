-- Function definitions example
-- Demonstrates defining and calling functions

-- Double a number
double x = x + x

-- Square a number
square x = x * x

-- Apply a function to a value
apply f x = f x

-- Compose: double(square(3)) = double(9) = 18
result = apply double (square 3)

main = print result  -- 18
