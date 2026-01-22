{-# LANGUAGE MagicHash #-}
{-# LANGUAGE UnboxedTuples #-}
{-# LANGUAGE BangPatterns #-}

-- |
-- Module      : BHC.Prelude
-- Description : Core types and functions for BHC
-- Copyright   : (c) BHC Contributors, 2026
-- License     : BSD-3-Clause
-- Maintainer  : bhc@example.com
-- Stability   : stable
--
-- The BHC Prelude provides core types and functions that are implicitly
-- imported into every BHC module (unless @NoImplicitPrelude@ is used).
--
-- = Overview
--
-- This module re-exports the essential types and functions from:
--
-- * "BHC.Data.Bool" - Boolean type and operations
-- * "BHC.Data.Maybe" - Optional values
-- * "BHC.Data.Either" - Disjoint unions
-- * "BHC.Data.Ordering" - Comparison results
-- * "BHC.Data.Tuple" - Tuple operations
-- * "BHC.Data.List" - List operations
-- * "BHC.Data.Function" - Function combinators
-- * "BHC.Control.Monad" - Monadic operations
--
-- = Differences from GHC Prelude
--
-- * All list operations participate in fusion
-- * Strict variants are preferred in Numeric Profile
-- * SIMD-accelerated numeric operations
--
-- = Usage
--
-- The Prelude is imported automatically:
--
-- @
-- module MyModule where
--
-- -- Prelude functions available without explicit import
-- main = print (sum [1..100])
-- @

module BHC.Prelude
  ( -- * Basic types
    -- ** Bool
    Bool(..)
  , (&&), (||), not, otherwise

    -- ** Maybe
  , Maybe(..)
  , maybe, isJust, isNothing, fromJust, fromMaybe
  , listToMaybe, maybeToList, catMaybes, mapMaybe

    -- ** Either
  , Either(..)
  , either, isLeft, isRight
  , fromLeft, fromRight
  , lefts, rights, partitionEithers

    -- ** Ordering
  , Ordering(..)
  , compare, (<), (<=), (>=), (>), max, min

    -- ** Tuples
  , fst, snd, curry, uncurry

    -- * Basic type classes
    -- ** Eq
  , Eq(..)

    -- ** Ord
  , Ord(..)

    -- ** Enum
  , Enum(..)

    -- ** Bounded
  , Bounded(..)

    -- ** Num
  , Num(..)

    -- ** Real
  , Real(..)

    -- ** Integral
  , Integral(..)

    -- ** Fractional
  , Fractional(..)

    -- ** Floating
  , Floating(..)

    -- ** RealFrac
  , RealFrac(..)

    -- ** RealFloat
  , RealFloat(..)

    -- ** Show
  , Show(..)

    -- ** Read
  , Read(..)

    -- * Functor, Applicative, Monad
  , Functor(..)
  , (<$>), (<$), ($>)
  , Applicative(..)
  , (<*>), (*>), (<*)
  , Monad(..)
  , (>>=), (>>), return, fail
  , (=<<), (>=>), (<=<)

    -- * Foldable and Traversable
  , Foldable(..)
  , Traversable(..)

    -- * Semigroup and Monoid
  , Semigroup(..)
  , Monoid(..)

    -- * List operations
    -- ** Basic
  , map, (++), filter
  , head, last, tail, init, null, length, (!!)

    -- ** Reducing
  , foldl, foldl', foldl1, foldr, foldr1
  , and, or, any, all
  , sum, product
  , concat, concatMap
  , maximum, minimum

    -- ** Building
  , scanl, scanl1, scanr, scanr1
  , iterate, repeat, replicate, cycle

    -- ** Sublists
  , take, drop, splitAt
  , takeWhile, dropWhile, span, break

    -- ** Searching
  , elem, notElem, lookup

    -- ** Zipping
  , zip, zip3, zipWith, zipWith3
  , unzip, unzip3

    -- ** String functions
  , lines, words, unlines, unwords

    -- * Numeric functions
  , subtract, even, odd, gcd, lcm
  , (^), (^^)
  , fromIntegral, realToFrac

    -- * Function combinators
  , id, const, (.), flip, ($), (&)
  , until, asTypeOf, error, undefined

    -- * I/O operations
  , IO
  , putChar, putStr, putStrLn, print
  , getChar, getLine, getContents
  , interact
  , readFile, writeFile, appendFile

    -- * Special types
  , FilePath
  , String
  , Char
  , Int, Integer, Float, Double
  , Rational
  , Word

  ) where

-- Implementation note: This file serves as the specification for what
-- the Prelude should contain. Actual implementations are in the
-- respective submodules and backed by Rust primitives where appropriate.

-- Core types
data Bool = False | True
data Maybe a = Nothing | Just a
data Either a b = Left a | Right b
data Ordering = LT | EQ | GT

-- Placeholder type class definitions
-- In actual implementation, these would be compiler builtins

class Eq a where
  (==), (/=) :: a -> a -> Bool
  x == y = not (x /= y)
  x /= y = not (x == y)
  {-# MINIMAL (==) | (/=) #-}

class Eq a => Ord a where
  compare :: a -> a -> Ordering
  (<), (<=), (>), (>=) :: a -> a -> Bool
  max, min :: a -> a -> a

  compare x y
    | x == y    = EQ
    | x <= y    = LT
    | otherwise = GT

  x <  y = compare x y == LT
  x <= y = compare x y /= GT
  x >  y = compare x y == GT
  x >= y = compare x y /= LT

  max x y = if x >= y then x else y
  min x y = if x <= y then x else y
  {-# MINIMAL compare | (<=) #-}

class Enum a where
  succ, pred :: a -> a
  toEnum :: Int -> a
  fromEnum :: a -> Int
  enumFrom :: a -> [a]
  enumFromThen :: a -> a -> [a]
  enumFromTo :: a -> a -> [a]
  enumFromThenTo :: a -> a -> a -> [a]

class Bounded a where
  minBound, maxBound :: a

class Num a where
  (+), (-), (*) :: a -> a -> a
  negate :: a -> a
  abs :: a -> a
  signum :: a -> a
  fromInteger :: Integer -> a

  x - y = x + negate y
  negate x = 0 - x
  {-# MINIMAL (+), (*), abs, signum, fromInteger, (negate | (-)) #-}

class Show a where
  show :: a -> String
  showsPrec :: Int -> a -> String -> String
  showList :: [a] -> String -> String

class Read a where
  readsPrec :: Int -> String -> [(a, String)]
  readList :: String -> [([a], String)]

class Functor f where
  fmap :: (a -> b) -> f a -> f b
  (<$) :: a -> f b -> f a
  (<$) = fmap . const

class Functor f => Applicative f where
  pure :: a -> f a
  (<*>) :: f (a -> b) -> f a -> f b
  (*>) :: f a -> f b -> f b
  a1 *> a2 = (id <$ a1) <*> a2
  (<*) :: f a -> f b -> f a
  (<*) = liftA2 const
  liftA2 :: (a -> b -> c) -> f a -> f b -> f c
  liftA2 f x = (<*>) (fmap f x)
  {-# MINIMAL pure, ((<*>) | liftA2) #-}

class Applicative m => Monad m where
  (>>=) :: m a -> (a -> m b) -> m b
  (>>) :: m a -> m b -> m b
  m >> k = m >>= \_ -> k
  return :: a -> m a
  return = pure
  {-# MINIMAL (>>=) #-}

class Semigroup a where
  (<>) :: a -> a -> a

class Semigroup a => Monoid a where
  mempty :: a
  mappend :: a -> a -> a
  mappend = (<>)
  mconcat :: [a] -> a
  mconcat = foldr mappend mempty

class Foldable t where
  fold :: Monoid m => t m -> m
  foldMap :: Monoid m => (a -> m) -> t a -> m
  foldMap' :: Monoid m => (a -> m) -> t a -> m
  foldr :: (a -> b -> b) -> b -> t a -> b
  foldr' :: (a -> b -> b) -> b -> t a -> b
  foldl :: (b -> a -> b) -> b -> t a -> b
  foldl' :: (b -> a -> b) -> b -> t a -> b
  foldr1 :: (a -> a -> a) -> t a -> a
  foldl1 :: (a -> a -> a) -> t a -> a
  toList :: t a -> [a]
  null :: t a -> Bool
  length :: t a -> Int
  elem :: Eq a => a -> t a -> Bool
  maximum :: Ord a => t a -> a
  minimum :: Ord a => t a -> a
  sum :: Num a => t a -> a
  product :: Num a => t a -> a
  {-# MINIMAL foldMap | foldr #-}

class (Functor t, Foldable t) => Traversable t where
  traverse :: Applicative f => (a -> f b) -> t a -> f (t b)
  sequenceA :: Applicative f => t (f a) -> f (t a)
  mapM :: Monad m => (a -> m b) -> t a -> m (t b)
  sequence :: Monad m => t (m a) -> m (t a)
  {-# MINIMAL traverse | sequenceA #-}

-- Core functions (implementations are primitives)

-- | Identity function
id :: a -> a
id x = x

-- | Constant function
const :: a -> b -> a
const x _ = x

-- | Function composition
(.) :: (b -> c) -> (a -> b) -> a -> c
(f . g) x = f (g x)

-- | Flip function arguments
flip :: (a -> b -> c) -> b -> a -> c
flip f y x = f x y

-- | Application operator (low precedence)
($) :: (a -> b) -> a -> b
f $ x = f x
infixr 0 $

-- | Reverse application operator
(&) :: a -> (a -> b) -> b
x & f = f x
infixl 1 &

-- | Boolean negation
not :: Bool -> Bool
not True = False
not False = True

-- | Boolean AND (short-circuit)
(&&) :: Bool -> Bool -> Bool
True && x = x
False && _ = False
infixr 3 &&

-- | Boolean OR (short-circuit)
(||) :: Bool -> Bool -> Bool
True || _ = True
False || x = x
infixr 2 ||

-- | Constant True (for guards)
otherwise :: Bool
otherwise = True

-- | Undefined value (throws error when evaluated)
undefined :: a
undefined = error "Prelude.undefined"

-- | Raise an error with message
error :: String -> a
error s = errorWithoutStackTrace s

-- Primitive - implemented in RTS
errorWithoutStackTrace :: String -> a
errorWithoutStackTrace = primError

foreign import ccall unsafe "bhc_error"
  primError :: String -> a

-- | Apply function until predicate is satisfied
until :: (a -> Bool) -> (a -> a) -> a -> a
until p f = go
  where
    go !x
      | p x       = x
      | otherwise = go (f x)

-- | Type-restricted identity
asTypeOf :: a -> a -> a
asTypeOf = const

-- Type aliases
type String = [Char]
type FilePath = String
type Rational = Ratio Integer

-- Placeholder for Ratio type
data Ratio a = !a :% !a

-- Placeholder IO type
newtype IO a = IO (State# RealWorld -> (# State# RealWorld, a #))

-- Placeholder primitive types
data State# s
data RealWorld

-- FFI declarations for primitives
foreign import ccall unsafe "bhc_putChar" primPutChar :: Char -> IO ()
foreign import ccall unsafe "bhc_getChar" primGetChar :: IO Char
foreign import ccall unsafe "bhc_putStr" primPutStr :: String -> IO ()

putChar :: Char -> IO ()
putChar = primPutChar

putStr :: String -> IO ()
putStr = primPutStr

putStrLn :: String -> IO ()
putStrLn s = putStr s >> putChar '\n'

print :: Show a => a -> IO ()
print x = putStrLn (show x)

getChar :: IO Char
getChar = primGetChar

getLine :: IO String
getLine = do
  c <- getChar
  if c == '\n'
    then return []
    else do
      cs <- getLine
      return (c:cs)

getContents :: IO String
getContents = undefined  -- Implemented in IO module

interact :: (String -> String) -> IO ()
interact f = do
  s <- getContents
  putStr (f s)

readFile :: FilePath -> IO String
readFile = undefined  -- Implemented in IO module

writeFile :: FilePath -> String -> IO ()
writeFile = undefined  -- Implemented in IO module

appendFile :: FilePath -> String -> IO ()
appendFile = undefined  -- Implemented in IO module

-- Maybe functions
maybe :: b -> (a -> b) -> Maybe a -> b
maybe n _ Nothing = n
maybe _ f (Just x) = f x

isJust :: Maybe a -> Bool
isJust (Just _) = True
isJust Nothing = False

isNothing :: Maybe a -> Bool
isNothing = not . isJust

fromJust :: Maybe a -> a
fromJust (Just x) = x
fromJust Nothing = error "Maybe.fromJust: Nothing"

fromMaybe :: a -> Maybe a -> a
fromMaybe d Nothing = d
fromMaybe _ (Just x) = x

-- Either functions
either :: (a -> c) -> (b -> c) -> Either a b -> c
either f _ (Left x) = f x
either _ g (Right y) = g y

isLeft :: Either a b -> Bool
isLeft (Left _) = True
isLeft _ = False

isRight :: Either a b -> Bool
isRight (Right _) = True
isRight _ = False

-- Tuple functions
fst :: (a, b) -> a
fst (x, _) = x

snd :: (a, b) -> b
snd (_, y) = y

curry :: ((a, b) -> c) -> a -> b -> c
curry f x y = f (x, y)

uncurry :: (a -> b -> c) -> (a, b) -> c
uncurry f (x, y) = f x y

-- Numeric functions
subtract :: Num a => a -> a -> a
subtract x y = y - x

even :: Integral a => a -> Bool
even n = n `rem` 2 == 0

odd :: Integral a => a -> Bool
odd = not . even

gcd :: Integral a => a -> a -> a
gcd x y = gcd' (abs x) (abs y)
  where
    gcd' a 0 = a
    gcd' a b = gcd' b (a `rem` b)

lcm :: Integral a => a -> a -> a
lcm _ 0 = 0
lcm 0 _ = 0
lcm x y = abs ((x `quot` gcd x y) * y)

-- List functions (placeholders - actual implementations use fusion)
map :: (a -> b) -> [a] -> [b]
map _ [] = []
map f (x:xs) = f x : map f xs
{-# NOINLINE [1] map #-}

filter :: (a -> Bool) -> [a] -> [a]
filter _ [] = []
filter p (x:xs)
  | p x       = x : filter p xs
  | otherwise = filter p xs
{-# NOINLINE [1] filter #-}

(++) :: [a] -> [a] -> [a]
[] ++ ys = ys
(x:xs) ++ ys = x : (xs ++ ys)
{-# NOINLINE [1] (++) #-}

-- Fusion rules would be defined here
{-# RULES
"map/map" forall f g xs. map f (map g xs) = map (f . g) xs
"map/filter" forall f p xs. map f (filter p xs) = foldr (\x acc -> if p x then f x : acc else acc) [] xs
  #-}
