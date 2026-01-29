fib 0 = 0
fib 1 = 1
fib n = fib (n - 1) + fib (n - 2)

main = print (fib 0) >> print (fib 1) >> print (fib 5) >> print (fib 10)
