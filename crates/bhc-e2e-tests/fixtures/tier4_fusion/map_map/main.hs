-- Test map/map fusion (should fuse into single traversal)
main = print (sum (map (*2) (map (+1) [1..10])))
