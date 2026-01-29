isEven n = if n == 0 then 1 else isOdd (n - 1)
isOdd n = if n == 0 then 0 else isEven (n - 1)

main = print (isEven 10) >> print (isOdd 7) >> print (isEven 3)
