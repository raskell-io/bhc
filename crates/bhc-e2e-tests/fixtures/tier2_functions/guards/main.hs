classify x = if x > 0 then 1 else if x < 0 then -1 else 0

main = print (classify 5) >> print (classify (-3)) >> print (classify 0)
