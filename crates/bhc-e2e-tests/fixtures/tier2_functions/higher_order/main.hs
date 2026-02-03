apply f x = f x
compose f g x = f (g x)

main = do
  print (apply (\x -> x + 1) 5)
  print (compose (\x -> x * 2) (\x -> x + 3) 4)
