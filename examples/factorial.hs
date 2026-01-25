-- Factorial example
-- Demonstrates recursion

factorial n = if n <= 1 then 1 else n * factorial (n - 1)

main = print (factorial 10)  -- 3628800
