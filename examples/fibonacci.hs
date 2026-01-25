-- Fibonacci sequence example
-- Demonstrates recursion

fib n = if n <= 1 then n else fib (n - 1) + fib (n - 2)

main = print (fib 15)  -- 610
