data Color = Red | Green | Blue

colorCode Red = 1
colorCode Green = 2
colorCode Blue = 3

main = print (colorCode Red) >> print (colorCode Green) >> print (colorCode Blue)
