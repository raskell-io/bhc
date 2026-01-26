-- |
-- Module      : BHC.Prelude
-- Description : The BHC Prelude - core types and functions
-- Copyright   : (c) BHC Contributors, 2026
-- License     : BSD-3-Clause
-- Stability   : stable
--
-- The BHC Prelude provides the core types and functions that are
-- automatically available in every BHC module.

{-# LANGUAGE NoImplicitPrelude #-}
{-# LANGUAGE MagicHash #-}
{-# LANGUAGE BangPatterns #-}

module BHC.Prelude (
    -- * Basic Types
    Bool(..), (&&), (||), not, otherwise, bool,
    Maybe(..), maybe, isJust, isNothing, fromMaybe, fromJust,
    Either(..), either, isLeft, isRight,
    Ordering(..),
    
    -- * Tuples
    fst, snd, curry, uncurry, swap,
    
    -- * Type Classes
    Eq(..), Ord(..), Show(..), Read(..),
    Enum(..), Bounded(..),
    Num(..), Integral(..), Fractional(..), Floating(..),
    Real(..), RealFrac(..), RealFloat(..),
    Semigroup(..), Monoid(..),
    Functor(..), (<$>), (<$), ($>), void,
    Applicative(..), liftA, liftA2, liftA3,
    Monad(..), (=<<), (>=>), (<=<), join, ap,
    Alternative(..), MonadPlus(..),
    Foldable(..), Traversable(..),
    
    -- * List Operations
    map, (++), filter, head, last, tail, init,
    null, length, (!!), reverse,
    take, drop, splitAt, takeWhile, dropWhile, span, break,
    elem, notElem, lookup, find,
    zip, zip3, zipWith, zipWith3, unzip, unzip3,
    iterate, repeat, replicate, cycle,
    foldr, foldl, foldl', foldr1, foldl1,
    and, or, any, all, sum, product, concat, concatMap,
    maximum, minimum, scanl, scanr,
    lines, words, unlines, unwords,
    sort, sortBy, sortOn,
    nub, delete, union, intersect, (\\),
    
    -- * Function Combinators
    id, const, (.), flip, ($), (&), on, fix,
    seq, ($!),
    
    -- * Numeric
    subtract, even, odd, gcd, lcm,
    (^), (^^), fromIntegral, realToFrac,
    
    -- * Monadic
    mapM, mapM_, forM, forM_, sequence, sequence_,
    (>>), (>>=), return, fail,
    when, unless, guard,
    filterM, foldM, foldM_, replicateM, replicateM_,
    
    -- * IO
    IO, putChar, putStr, putStrLn, print,
    getChar, getLine, getContents,
    readFile, writeFile, appendFile, interact,
    
    -- * Error
    error, undefined, errorWithoutStackTrace,
    
    -- * Types
    Char, String, Int, Integer, Float, Double, Word,
    Rational, FilePath,
) where

-- Primitive types from runtime
import GHC.Prim
import GHC.Types hiding (Module)

-- ============================================================
-- Primitive Operations (FFI)
-- ============================================================

-- Int primitives
foreign import ccall unsafe "bhc_eq_int" primEqInt :: Int -> Int -> Bool
foreign import ccall unsafe "bhc_compare_int" primCompareInt :: Int -> Int -> Ordering
foreign import ccall unsafe "bhc_lt_int" primLtInt :: Int -> Int -> Bool
foreign import ccall unsafe "bhc_le_int" primLeInt :: Int -> Int -> Bool
foreign import ccall unsafe "bhc_gt_int" primGtInt :: Int -> Int -> Bool
foreign import ccall unsafe "bhc_ge_int" primGeInt :: Int -> Int -> Bool
foreign import ccall unsafe "bhc_add_int" primAddInt :: Int -> Int -> Int
foreign import ccall unsafe "bhc_sub_int" primSubInt :: Int -> Int -> Int
foreign import ccall unsafe "bhc_mul_int" primMulInt :: Int -> Int -> Int
foreign import ccall unsafe "bhc_negate_int" primNegateInt :: Int -> Int
foreign import ccall unsafe "bhc_quot_int" primQuotInt :: Int -> Int -> Int
foreign import ccall unsafe "bhc_rem_int" primRemInt :: Int -> Int -> Int
foreign import ccall unsafe "bhc_div_int" primDivInt :: Int -> Int -> Int
foreign import ccall unsafe "bhc_mod_int" primModInt :: Int -> Int -> Int
foreign import ccall unsafe "bhc_min_int" primMinInt :: Int
foreign import ccall unsafe "bhc_max_int" primMaxInt :: Int
foreign import ccall unsafe "bhc_int_to_integer" primIntToInteger :: Int -> Integer
foreign import ccall unsafe "bhc_integer_to_int" primIntegerToInt :: Integer -> Int
foreign import ccall unsafe "bhc_show_int" primShowInt :: Int -> String
foreign import ccall unsafe "bhc_reads_prec_int" primReadsPrecInt :: Int -> ReadS Int

-- Float primitives
foreign import ccall unsafe "bhc_eq_float" primEqFloat :: Float -> Float -> Bool
foreign import ccall unsafe "bhc_compare_float" primCompareFloat :: Float -> Float -> Ordering
foreign import ccall unsafe "bhc_lt_float" primLtFloat :: Float -> Float -> Bool
foreign import ccall unsafe "bhc_le_float" primLeFloat :: Float -> Float -> Bool
foreign import ccall unsafe "bhc_gt_float" primGtFloat :: Float -> Float -> Bool
foreign import ccall unsafe "bhc_ge_float" primGeFloat :: Float -> Float -> Bool
foreign import ccall unsafe "bhc_add_float" primAddFloat :: Float -> Float -> Float
foreign import ccall unsafe "bhc_sub_float" primSubFloat :: Float -> Float -> Float
foreign import ccall unsafe "bhc_mul_float" primMulFloat :: Float -> Float -> Float
foreign import ccall unsafe "bhc_div_float" primDivFloat :: Float -> Float -> Float
foreign import ccall unsafe "bhc_negate_float" primNegateFloat :: Float -> Float
foreign import ccall unsafe "bhc_abs_float" primAbsFloat :: Float -> Float
foreign import ccall unsafe "bhc_integer_to_float" primIntegerToFloat :: Integer -> Float
foreign import ccall unsafe "bhc_rational_to_float" primRationalToFloat :: Rational -> Float
foreign import ccall unsafe "bhc_float_to_rational" primFloatToRational :: Float -> Rational
foreign import ccall unsafe "bhc_exp_float" primExpFloat :: Float -> Float
foreign import ccall unsafe "bhc_log_float" primLogFloat :: Float -> Float
foreign import ccall unsafe "bhc_sqrt_float" primSqrtFloat :: Float -> Float
foreign import ccall unsafe "bhc_pow_float" primPowFloat :: Float -> Float -> Float
foreign import ccall unsafe "bhc_sin_float" primSinFloat :: Float -> Float
foreign import ccall unsafe "bhc_cos_float" primCosFloat :: Float -> Float
foreign import ccall unsafe "bhc_tan_float" primTanFloat :: Float -> Float
foreign import ccall unsafe "bhc_asin_float" primAsinFloat :: Float -> Float
foreign import ccall unsafe "bhc_acos_float" primAcosFloat :: Float -> Float
foreign import ccall unsafe "bhc_atan_float" primAtanFloat :: Float -> Float
foreign import ccall unsafe "bhc_sinh_float" primSinhFloat :: Float -> Float
foreign import ccall unsafe "bhc_cosh_float" primCoshFloat :: Float -> Float
foreign import ccall unsafe "bhc_tanh_float" primTanhFloat :: Float -> Float
foreign import ccall unsafe "bhc_asinh_float" primAsinhFloat :: Float -> Float
foreign import ccall unsafe "bhc_acosh_float" primAcoshFloat :: Float -> Float
foreign import ccall unsafe "bhc_atanh_float" primAtanhFloat :: Float -> Float
foreign import ccall unsafe "bhc_proper_fraction_float" primProperFractionFloat :: Integral b => Float -> (b, Float)
foreign import ccall unsafe "bhc_truncate_float" primTruncateFloat :: Integral b => Float -> b
foreign import ccall unsafe "bhc_round_float" primRoundFloat :: Integral b => Float -> b
foreign import ccall unsafe "bhc_ceiling_float" primCeilingFloat :: Integral b => Float -> b
foreign import ccall unsafe "bhc_floor_float" primFloorFloat :: Integral b => Float -> b
foreign import ccall unsafe "bhc_show_float" primShowFloat :: Float -> String
foreign import ccall unsafe "bhc_reads_prec_float" primReadsPrecFloat :: Int -> ReadS Float

-- Double primitives
foreign import ccall unsafe "bhc_eq_double" primEqDouble :: Double -> Double -> Bool
foreign import ccall unsafe "bhc_compare_double" primCompareDouble :: Double -> Double -> Ordering
foreign import ccall unsafe "bhc_lt_double" primLtDouble :: Double -> Double -> Bool
foreign import ccall unsafe "bhc_le_double" primLeDouble :: Double -> Double -> Bool
foreign import ccall unsafe "bhc_gt_double" primGtDouble :: Double -> Double -> Bool
foreign import ccall unsafe "bhc_ge_double" primGeDouble :: Double -> Double -> Bool
foreign import ccall unsafe "bhc_add_double" primAddDouble :: Double -> Double -> Double
foreign import ccall unsafe "bhc_sub_double" primSubDouble :: Double -> Double -> Double
foreign import ccall unsafe "bhc_mul_double" primMulDouble :: Double -> Double -> Double
foreign import ccall unsafe "bhc_div_double" primDivDouble :: Double -> Double -> Double
foreign import ccall unsafe "bhc_negate_double" primNegateDouble :: Double -> Double
foreign import ccall unsafe "bhc_abs_double" primAbsDouble :: Double -> Double
foreign import ccall unsafe "bhc_integer_to_double" primIntegerToDouble :: Integer -> Double
foreign import ccall unsafe "bhc_rational_to_double" primRationalToDouble :: Rational -> Double
foreign import ccall unsafe "bhc_double_to_rational" primDoubleToRational :: Double -> Rational
foreign import ccall unsafe "bhc_exp_double" primExpDouble :: Double -> Double
foreign import ccall unsafe "bhc_log_double" primLogDouble :: Double -> Double
foreign import ccall unsafe "bhc_sqrt_double" primSqrtDouble :: Double -> Double
foreign import ccall unsafe "bhc_pow_double" primPowDouble :: Double -> Double -> Double
foreign import ccall unsafe "bhc_sin_double" primSinDouble :: Double -> Double
foreign import ccall unsafe "bhc_cos_double" primCosDouble :: Double -> Double
foreign import ccall unsafe "bhc_tan_double" primTanDouble :: Double -> Double
foreign import ccall unsafe "bhc_asin_double" primAsinDouble :: Double -> Double
foreign import ccall unsafe "bhc_acos_double" primAcosDouble :: Double -> Double
foreign import ccall unsafe "bhc_atan_double" primAtanDouble :: Double -> Double
foreign import ccall unsafe "bhc_sinh_double" primSinhDouble :: Double -> Double
foreign import ccall unsafe "bhc_cosh_double" primCoshDouble :: Double -> Double
foreign import ccall unsafe "bhc_tanh_double" primTanhDouble :: Double -> Double
foreign import ccall unsafe "bhc_asinh_double" primAsinhDouble :: Double -> Double
foreign import ccall unsafe "bhc_acosh_double" primAcoshDouble :: Double -> Double
foreign import ccall unsafe "bhc_atanh_double" primAtanhDouble :: Double -> Double
foreign import ccall unsafe "bhc_proper_fraction_double" primProperFractionDouble :: Integral b => Double -> (b, Double)
foreign import ccall unsafe "bhc_truncate_double" primTruncateDouble :: Integral b => Double -> b
foreign import ccall unsafe "bhc_round_double" primRoundDouble :: Integral b => Double -> b
foreign import ccall unsafe "bhc_ceiling_double" primCeilingDouble :: Integral b => Double -> b
foreign import ccall unsafe "bhc_floor_double" primFloorDouble :: Integral b => Double -> b
foreign import ccall unsafe "bhc_show_double" primShowDouble :: Double -> String
foreign import ccall unsafe "bhc_reads_prec_double" primReadsPrecDouble :: Int -> ReadS Double

-- Char primitives
foreign import ccall unsafe "bhc_eq_char" primEqChar :: Char -> Char -> Bool
foreign import ccall unsafe "bhc_compare_char" primCompareChar :: Char -> Char -> Ordering
foreign import ccall unsafe "bhc_lt_char" primLtChar :: Char -> Char -> Bool
foreign import ccall unsafe "bhc_le_char" primLeChar :: Char -> Char -> Bool
foreign import ccall unsafe "bhc_gt_char" primGtChar :: Char -> Char -> Bool
foreign import ccall unsafe "bhc_ge_char" primGeChar :: Char -> Char -> Bool
foreign import ccall unsafe "bhc_char_to_int" primCharToInt :: Char -> Int
foreign import ccall unsafe "bhc_int_to_char" primIntToChar :: Int -> Char
foreign import ccall unsafe "bhc_show_char" primShowChar :: Char -> String
foreign import ccall unsafe "bhc_reads_prec_char" primReadsPrecChar :: Int -> ReadS Char

-- Rational type (for numeric operations)
type Rational = Ratio Integer

data Ratio a = !a :% !a
infixl 7 :%

(%) :: Integral a => a -> a -> Ratio a
x % y = reduce (x * signum y) (abs y)
  where
    reduce a b = let g = gcd a b in (a `quot` g) :% (b `quot` g)
infixl 7 %

numerator, denominator :: Ratio a -> a
numerator (x :% _) = x
denominator (_ :% y) = y

instance Integral a => Eq (Ratio a) where
    (x :% y) == (x' :% y') = x == x' && y == y'

instance Integral a => Ord (Ratio a) where
    (x :% y) `compare` (x' :% y') = compare (x * y') (x' * y)

instance Integral a => Num (Ratio a) where
    (x :% y) + (x' :% y') = (x * y' + x' * y) % (y * y')
    (x :% y) - (x' :% y') = (x * y' - x' * y) % (y * y')
    (x :% y) * (x' :% y') = (x * x') % (y * y')
    negate (x :% y) = negate x :% y
    abs (x :% y) = abs x :% y
    signum (x :% _) = signum x :% 1
    fromInteger x = fromInteger x :% 1

instance Integral a => Fractional (Ratio a) where
    (x :% y) / (x' :% y') = (x * y') % (y * x')
    recip (x :% y) = y % x
    fromRational (x :% y) = fromInteger x :% fromInteger y

instance (Integral a, Show a) => Show (Ratio a) where
    show (x :% y) = show x ++ " % " ++ show y

-- | Boolean type
data Bool = False | True
    deriving (Eq, Ord, Show, Read, Enum, Bounded)

-- | Boolean and (short-circuit)
(&&) :: Bool -> Bool -> Bool
True  && x = x
False && _ = False
infixr 3 &&

-- | Boolean or (short-circuit)
(||) :: Bool -> Bool -> Bool
True  || _ = True
False || x = x
infixr 2 ||

-- | Boolean negation
not :: Bool -> Bool
not True  = False
not False = True

-- | Always True
otherwise :: Bool
otherwise = True

-- | Case analysis for Bool
bool :: a -> a -> Bool -> a
bool f _ False = f
bool _ t True  = t

-- | The 'Maybe' type encapsulates an optional value.
--
-- A value of type @Maybe a@ either contains a value of type @a@
-- (represented as @Just a@), or it is empty (represented as @Nothing@).
--
-- Using 'Maybe' is a good way to deal with errors or exceptional cases
-- without resorting to drastic measures such as 'error'.
--
-- ==== __Examples__
--
-- >>> Just 42
-- Just 42
--
-- >>> Nothing :: Maybe Int
-- Nothing
--
-- >>> fmap (+1) (Just 41)
-- Just 42
data Maybe a
    = Nothing  -- ^ No value
    | Just a   -- ^ A value of type @a@
    deriving (Eq, Ord, Show, Read)

instance Functor Maybe where
    fmap _ Nothing  = Nothing
    fmap f (Just x) = Just (f x)

instance Applicative Maybe where
    pure = Just
    Nothing <*> _ = Nothing
    Just f <*> x  = fmap f x

instance Monad Maybe where
    Nothing >>= _ = Nothing
    Just x  >>= f = f x

-- | The 'maybe' function takes a default value, a function, and a 'Maybe'
-- value. If the 'Maybe' value is 'Nothing', the function returns the default
-- value. Otherwise, it applies the function to the value inside the 'Just'
-- and returns the result.
--
-- ==== __Examples__
--
-- >>> maybe 0 (+1) (Just 41)
-- 42
--
-- >>> maybe 0 (+1) Nothing
-- 0
maybe :: b -> (a -> b) -> Maybe a -> b
maybe n _ Nothing  = n
maybe _ f (Just x) = f x

-- | Returns 'True' iff the argument is of the form @Just _@.
--
-- ==== __Examples__
--
-- >>> isJust (Just 42)
-- True
--
-- >>> isJust Nothing
-- False
isJust :: Maybe a -> Bool
isJust (Just _) = True
isJust Nothing  = False

-- | Returns 'True' iff the argument is 'Nothing'.
--
-- ==== __Examples__
--
-- >>> isNothing Nothing
-- True
--
-- >>> isNothing (Just 42)
-- False
isNothing :: Maybe a -> Bool
isNothing = not . isJust

-- | The 'fromMaybe' function takes a default value and a 'Maybe' value.
-- If the 'Maybe' is 'Nothing', it returns the default value; otherwise,
-- it returns the value contained in the 'Maybe'.
--
-- ==== __Examples__
--
-- >>> fromMaybe 0 (Just 42)
-- 42
--
-- >>> fromMaybe 0 Nothing
-- 0
fromMaybe :: a -> Maybe a -> a
fromMaybe d Nothing  = d
fromMaybe _ (Just x) = x

-- | Extract the value from a 'Just', throwing an error if the argument
-- is 'Nothing'.
--
-- __Warning__: This is a partial function. Prefer 'fromMaybe' or pattern
-- matching when possible.
--
-- ==== __Examples__
--
-- >>> fromJust (Just 42)
-- 42
fromJust :: Maybe a -> a
fromJust (Just x) = x
fromJust Nothing  = error "fromJust: Nothing"

-- | The 'Either' type represents values with two possibilities: a value of
-- type @Either a b@ is either @Left a@ or @Right b@.
--
-- The 'Either' type is sometimes used to represent a value which is either
-- correct or an error; by convention, the 'Left' constructor is used to hold
-- an error value and the 'Right' constructor is used to hold a correct value
-- (mnemonic: \"right\" also means \"correct\").
--
-- ==== __Examples__
--
-- >>> Left "error" :: Either String Int
-- Left "error"
--
-- >>> Right 42 :: Either String Int
-- Right 42
--
-- >>> fmap (+1) (Right 41)
-- Right 42
data Either a b
    = Left a   -- ^ Typically an error value
    | Right b  -- ^ Typically the success value
    deriving (Eq, Ord, Show, Read)

instance Functor (Either a) where
    fmap _ (Left x)  = Left x
    fmap f (Right y) = Right (f y)

instance Applicative (Either a) where
    pure = Right
    Left x  <*> _ = Left x
    Right f <*> y = fmap f y

instance Monad (Either a) where
    Left x  >>= _ = Left x
    Right y >>= f = f y

-- | Case analysis for the 'Either' type. If the value is @Left a@, apply
-- the first function to @a@; if it is @Right b@, apply the second function
-- to @b@.
--
-- ==== __Examples__
--
-- >>> either length (*2) (Left "hello")
-- 5
--
-- >>> either length (*2) (Right 21)
-- 42
either :: (a -> c) -> (b -> c) -> Either a b -> c
either f _ (Left x)  = f x
either _ g (Right y) = g y

-- | Return 'True' if the given value is a 'Left'-value, 'False' otherwise.
--
-- ==== __Examples__
--
-- >>> isLeft (Left "error")
-- True
--
-- >>> isLeft (Right 42)
-- False
isLeft :: Either a b -> Bool
isLeft (Left _) = True
isLeft _        = False

-- | Return 'True' if the given value is a 'Right'-value, 'False' otherwise.
--
-- ==== __Examples__
--
-- >>> isRight (Right 42)
-- True
--
-- >>> isRight (Left "error")
-- False
isRight :: Either a b -> Bool
isRight (Right _) = True
isRight _         = False

-- | Comparison result
data Ordering = LT | EQ | GT
    deriving (Eq, Ord, Show, Read, Enum, Bounded)

-- -----------------------------------------------------------------------------
-- Tuple operations

-- | Extract the first component of a pair.
--
-- ==== __Examples__
--
-- >>> fst (1, "hello")
-- 1
fst :: (a, b) -> a
fst (x, _) = x

-- | Extract the second component of a pair.
--
-- ==== __Examples__
--
-- >>> snd (1, "hello")
-- "hello"
snd :: (a, b) -> b
snd (_, y) = y

-- | 'curry' converts an uncurried function to a curried function.
--
-- ==== __Examples__
--
-- >>> curry fst 1 2
-- 1
curry :: ((a, b) -> c) -> a -> b -> c
curry f x y = f (x, y)

-- | 'uncurry' converts a curried function to a function on pairs.
--
-- ==== __Examples__
--
-- >>> uncurry (+) (1, 2)
-- 3
uncurry :: (a -> b -> c) -> (a, b) -> c
uncurry f (x, y) = f x y

-- | Swap the components of a pair.
--
-- ==== __Examples__
--
-- >>> swap (1, "hello")
-- ("hello", 1)
swap :: (a, b) -> (b, a)
swap (x, y) = (y, x)

-- -----------------------------------------------------------------------------
-- Type classes

-- | The 'Eq' class defines equality ('==') and inequality ('/=').
--
-- All basic datatypes exported by the Prelude are instances of 'Eq',
-- and 'Eq' may be derived for any datatype whose constituents are also
-- instances of 'Eq'.
--
-- ==== __Minimal complete definition__
--
-- Either '==' or '/='.
class Eq a where
    -- | Equality test.
    (==) :: a -> a -> Bool
    -- | Inequality test.
    (/=) :: a -> a -> Bool
    x /= y = not (x == y)
    x == y = not (x /= y)
    {-# MINIMAL (==) | (/=) #-}

-- | The 'Ord' class is used for totally ordered datatypes.
--
-- Instances of 'Ord' can be derived for any user-defined datatype whose
-- constituent types are in 'Ord'. The declared order of the constructors
-- in the data declaration determines the ordering in derived 'Ord' instances.
--
-- ==== __Minimal complete definition__
--
-- Either 'compare' or '<='.
--
-- ==== __Laws__
--
-- * Transitivity: if @x <= y && y <= z@ then @x <= z@
-- * Reflexivity: @x <= x@
-- * Antisymmetry: if @x <= y && y <= x@ then @x == y@
class Eq a => Ord a where
    -- | Compare two values.
    compare :: a -> a -> Ordering
    -- | Less than.
    (<) :: a -> a -> Bool
    -- | Less than or equal.
    (<=) :: a -> a -> Bool
    -- | Greater than.
    (>) :: a -> a -> Bool
    -- | Greater than or equal.
    (>=) :: a -> a -> Bool
    -- | The larger of two values.
    max :: a -> a -> a
    -- | The smaller of two values.
    min :: a -> a -> a

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

class Show a where
    showsPrec :: Int -> a -> ShowS
    show :: a -> String
    showList :: [a] -> ShowS
    
    showsPrec _ x s = show x ++ s
    show x = showsPrec 0 x ""
    showList ls s = showList__ shows ls s
    {-# MINIMAL showsPrec | show #-}

type ShowS = String -> String

showList__ :: (a -> ShowS) -> [a] -> ShowS
showList__ _     []     s = "[]" ++ s
showList__ showx (x:xs) s = '[' : showx x (showl xs)
  where showl []     = ']' : s
        showl (y:ys) = ',' : showx y (showl ys)

shows :: Show a => a -> ShowS
shows = showsPrec 0

class Read a where
    readsPrec :: Int -> ReadS a
    readList :: ReadS [a]
    {-# MINIMAL readsPrec #-}

type ReadS a = String -> [(a, String)]

class Enum a where
    succ, pred :: a -> a
    toEnum :: Int -> a
    fromEnum :: a -> Int
    enumFrom :: a -> [a]
    enumFromThen :: a -> a -> [a]
    enumFromTo :: a -> a -> [a]
    enumFromThenTo :: a -> a -> a -> [a]
    
    succ = toEnum . (+1) . fromEnum
    pred = toEnum . subtract 1 . fromEnum
    enumFrom x = map toEnum [fromEnum x ..]
    enumFromThen x y = map toEnum [fromEnum x, fromEnum y ..]
    enumFromTo x y = map toEnum [fromEnum x .. fromEnum y]
    enumFromThenTo x1 x2 y = map toEnum [fromEnum x1, fromEnum x2 .. fromEnum y]
    {-# MINIMAL toEnum, fromEnum #-}

class Bounded a where
    minBound, maxBound :: a

class Num a where
    (+), (-), (*) :: a -> a -> a
    negate, abs, signum :: a -> a
    fromInteger :: Integer -> a
    
    x - y = x + negate y
    negate x = 0 - x
    {-# MINIMAL (+), (*), abs, signum, fromInteger, (negate | (-)) #-}

class (Num a, Ord a) => Real a where
    toRational :: a -> Rational

class (Real a, Enum a) => Integral a where
    quot, rem, div, mod :: a -> a -> a
    quotRem, divMod :: a -> a -> (a, a)
    toInteger :: a -> Integer
    {-# MINIMAL quotRem, toInteger #-}

class Num a => Fractional a where
    (/) :: a -> a -> a
    recip :: a -> a
    fromRational :: Rational -> a
    
    recip x = 1 / x
    {-# MINIMAL fromRational, (recip | (/)) #-}

class Fractional a => Floating a where
    pi :: a
    exp, log, sqrt :: a -> a
    (**), logBase :: a -> a -> a
    sin, cos, tan, asin, acos, atan :: a -> a
    sinh, cosh, tanh, asinh, acosh, atanh :: a -> a
    {-# MINIMAL pi, exp, log, sin, cos, asin, acos, atan, sinh, cosh, asinh, acosh, atanh #-}

class (Real a, Fractional a) => RealFrac a where
    properFraction :: Integral b => a -> (b, a)
    truncate, round, ceiling, floor :: Integral b => a -> b
    {-# MINIMAL properFraction #-}

class (RealFrac a, Floating a) => RealFloat a where
    floatRadix :: a -> Integer
    floatDigits :: a -> Int
    floatRange :: a -> (Int, Int)
    decodeFloat :: a -> (Integer, Int)
    encodeFloat :: Integer -> Int -> a
    isNaN, isInfinite, isDenormalized, isNegativeZero, isIEEE :: a -> Bool
    atan2 :: a -> a -> a
    {-# MINIMAL floatRadix, floatDigits, floatRange, decodeFloat, encodeFloat, isNaN, isInfinite, isDenormalized, isNegativeZero, isIEEE #-}

class Semigroup a where
    (<>) :: a -> a -> a

class Semigroup a => Monoid a where
    mempty :: a
    mappend :: a -> a -> a
    mconcat :: [a] -> a
    
    mappend = (<>)
    mconcat = foldr mappend mempty
    {-# MINIMAL mempty #-}

class Functor f where
    fmap :: (a -> b) -> f a -> f b
    (<$) :: a -> f b -> f a
    (<$) = fmap . const

(<$>) :: Functor f => (a -> b) -> f a -> f b
(<$>) = fmap
infixl 4 <$>

($>) :: Functor f => f a -> b -> f b
($>) = flip (<$)
infixl 4 $>

void :: Functor f => f a -> f ()
void = fmap (const ())

class Functor f => Applicative f where
    pure :: a -> f a
    (<*>) :: f (a -> b) -> f a -> f b
    (*>) :: f a -> f b -> f b
    (<*) :: f a -> f b -> f a
    liftA2 :: (a -> b -> c) -> f a -> f b -> f c
    
    a1 *> a2 = (id <$ a1) <*> a2
    a1 <* a2 = liftA2 const a1 a2
    liftA2 f x y = f <$> x <*> y
    {-# MINIMAL pure, ((<*>) | liftA2) #-}

liftA :: Applicative f => (a -> b) -> f a -> f b
liftA f a = pure f <*> a

liftA3 :: Applicative f => (a -> b -> c -> d) -> f a -> f b -> f c -> f d
liftA3 f a b c = liftA2 f a b <*> c

class Applicative m => Monad m where
    (>>=) :: m a -> (a -> m b) -> m b
    (>>) :: m a -> m b -> m b
    return :: a -> m a
    
    return = pure
    m >> k = m >>= \_ -> k
    {-# MINIMAL (>>=) #-}

(=<<) :: Monad m => (a -> m b) -> m a -> m b
f =<< x = x >>= f
infixr 1 =<<

(>=>) :: Monad m => (a -> m b) -> (b -> m c) -> a -> m c
(f >=> g) x = f x >>= g
infixr 1 >=>

(<=<) :: Monad m => (b -> m c) -> (a -> m b) -> a -> m c
(<=<) = flip (>=>)
infixr 1 <=<

join :: Monad m => m (m a) -> m a
join x = x >>= id

ap :: Monad m => m (a -> b) -> m a -> m b
ap mf mx = mf >>= \f -> mx >>= \x -> return (f x)

class Applicative f => Alternative f where
    empty :: f a
    (<|>) :: f a -> f a -> f a
    some :: f a -> f [a]
    many :: f a -> f [a]
    
    some v = (:) <$> v <*> many v
    many v = some v <|> pure []
    {-# MINIMAL empty, (<|>) #-}

class (Alternative m, Monad m) => MonadPlus m where
    mzero :: m a
    mplus :: m a -> m a -> m a
    mzero = empty
    mplus = (<|>)

class Foldable t where
    fold :: Monoid m => t m -> m
    foldMap :: Monoid m => (a -> m) -> t a -> m
    foldr :: (a -> b -> b) -> b -> t a -> b
    foldl :: (b -> a -> b) -> b -> t a -> b
    foldl' :: (b -> a -> b) -> b -> t a -> b
    toList :: t a -> [a]
    null :: t a -> Bool
    length :: t a -> Int
    elem :: Eq a => a -> t a -> Bool
    maximum :: Ord a => t a -> a
    minimum :: Ord a => t a -> a
    sum :: Num a => t a -> a
    product :: Num a => t a -> a
    
    fold = foldMap id
    foldMap f = foldr (mappend . f) mempty
    toList = foldr (:) []
    null = foldr (\_ _ -> False) True
    length = foldl' (\c _ -> c + 1) 0
    elem x = any (== x)
    {-# MINIMAL foldMap | foldr #-}

class (Functor t, Foldable t) => Traversable t where
    traverse :: Applicative f => (a -> f b) -> t a -> f (t b)
    sequenceA :: Applicative f => t (f a) -> f (t a)
    mapM :: Monad m => (a -> m b) -> t a -> m (t b)
    sequence :: Monad m => t (m a) -> m (t a)
    
    sequenceA = traverse id
    mapM = traverse
    sequence = sequenceA
    {-# MINIMAL traverse | sequenceA #-}

-- ============================================================
-- Primitive Type Instances
-- ============================================================

-- Int instances
instance Eq Int where
    (==) = primEqInt

instance Ord Int where
    compare = primCompareInt
    (<)  = primLtInt
    (<=) = primLeInt
    (>)  = primGtInt
    (>=) = primGeInt

instance Num Int where
    (+) = primAddInt
    (-) = primSubInt
    (*) = primMulInt
    negate = primNegateInt
    abs x = if x < 0 then negate x else x
    signum x = if x < 0 then (-1) else if x > 0 then 1 else 0
    fromInteger = primIntegerToInt

instance Real Int where
    toRational x = toInteger x % 1

instance Enum Int where
    toEnum = id
    fromEnum = id
    succ x = x + 1
    pred x = x - 1

instance Bounded Int where
    minBound = primMinInt
    maxBound = primMaxInt

instance Integral Int where
    quot = primQuotInt
    rem = primRemInt
    div = primDivInt
    mod = primModInt
    quotRem x y = (quot x y, rem x y)
    divMod x y = (div x y, mod x y)
    toInteger = primIntToInteger

instance Show Int where
    show = primShowInt

instance Read Int where
    readsPrec = primReadsPrecInt

-- Float instances
instance Eq Float where
    (==) = primEqFloat

instance Ord Float where
    compare = primCompareFloat
    (<)  = primLtFloat
    (<=) = primLeFloat
    (>)  = primGtFloat
    (>=) = primGeFloat

instance Num Float where
    (+) = primAddFloat
    (-) = primSubFloat
    (*) = primMulFloat
    negate = primNegateFloat
    abs = primAbsFloat
    signum x = if x < 0 then (-1) else if x > 0 then 1 else 0
    fromInteger = primIntegerToFloat

instance Fractional Float where
    (/) = primDivFloat
    recip x = 1 / x
    fromRational = primRationalToFloat

instance Floating Float where
    pi = 3.141592653589793
    exp = primExpFloat
    log = primLogFloat
    sqrt = primSqrtFloat
    (**) = primPowFloat
    logBase x y = log y / log x
    sin = primSinFloat
    cos = primCosFloat
    tan = primTanFloat
    asin = primAsinFloat
    acos = primAcosFloat
    atan = primAtanFloat
    sinh = primSinhFloat
    cosh = primCoshFloat
    tanh = primTanhFloat
    asinh = primAsinhFloat
    acosh = primAcoshFloat
    atanh = primAtanhFloat

instance Real Float where
    toRational = primFloatToRational

instance RealFrac Float where
    properFraction = primProperFractionFloat
    truncate = primTruncateFloat
    round = primRoundFloat
    ceiling = primCeilingFloat
    floor = primFloorFloat

instance Show Float where
    show = primShowFloat

instance Read Float where
    readsPrec = primReadsPrecFloat

-- Double instances
instance Eq Double where
    (==) = primEqDouble

instance Ord Double where
    compare = primCompareDouble
    (<)  = primLtDouble
    (<=) = primLeDouble
    (>)  = primGtDouble
    (>=) = primGeDouble

instance Num Double where
    (+) = primAddDouble
    (-) = primSubDouble
    (*) = primMulDouble
    negate = primNegateDouble
    abs = primAbsDouble
    signum x = if x < 0 then (-1) else if x > 0 then 1 else 0
    fromInteger = primIntegerToDouble

instance Fractional Double where
    (/) = primDivDouble
    recip x = 1 / x
    fromRational = primRationalToDouble

instance Floating Double where
    pi = 3.141592653589793238462643383279
    exp = primExpDouble
    log = primLogDouble
    sqrt = primSqrtDouble
    (**) = primPowDouble
    logBase x y = log y / log x
    sin = primSinDouble
    cos = primCosDouble
    tan = primTanDouble
    asin = primAsinDouble
    acos = primAcosDouble
    atan = primAtanDouble
    sinh = primSinhDouble
    cosh = primCoshDouble
    tanh = primTanhDouble
    asinh = primAsinhDouble
    acosh = primAcoshDouble
    atanh = primAtanhDouble

instance Real Double where
    toRational = primDoubleToRational

instance RealFrac Double where
    properFraction = primProperFractionDouble
    truncate = primTruncateDouble
    round = primRoundDouble
    ceiling = primCeilingDouble
    floor = primFloorDouble

instance Show Double where
    show = primShowDouble

instance Read Double where
    readsPrec = primReadsPrecDouble

-- Char instances
instance Eq Char where
    (==) = primEqChar

instance Ord Char where
    compare = primCompareChar
    (<)  = primLtChar
    (<=) = primLeChar
    (>)  = primGtChar
    (>=) = primGeChar

instance Enum Char where
    toEnum = primIntToChar
    fromEnum = primCharToInt
    enumFromTo x y = map toEnum [fromEnum x .. fromEnum y]

instance Bounded Char where
    minBound = '\0'
    maxBound = '\x10FFFF'

instance Show Char where
    show = primShowChar
    showList cs = ('"' :) . showLitString cs . ('"' :)

instance Read Char where
    readsPrec = primReadsPrecChar

-- Helper for showing string literals
showLitString :: String -> ShowS
showLitString []     s = s
showLitString (c:cs) s = showLitChar c (showLitString cs s)

showLitChar :: Char -> ShowS
showLitChar c s
    | c == '"'  = '\\' : '"' : s
    | c == '\\' = '\\' : '\\' : s
    | c >= ' ' && c <= '~' = c : s
    | otherwise = primShowChar c ++ s

-- List instance
instance Functor [] where
    fmap = map

instance Applicative [] where
    pure x = [x]
    fs <*> xs = [f x | f <- fs, x <- xs]

instance Monad [] where
    xs >>= f = concatMap f xs

instance Alternative [] where
    empty = []
    (<|>) = (++)

instance MonadPlus []

instance Semigroup [a] where
    (<>) = (++)

instance Monoid [a] where
    mempty = []

instance Foldable [] where
    foldr _ z []     = z
    foldr f z (x:xs) = f x (foldr f z xs)
    
    foldl _ z []     = z
    foldl f z (x:xs) = foldl f (f z x) xs
    
    foldl' _ z []     = z
    foldl' f z (x:xs) = let z' = f z x in z' `seq` foldl' f z' xs

instance Traversable [] where
    traverse _ []     = pure []
    traverse f (x:xs) = (:) <$> f x <*> traverse f xs

-- Function combinators
id :: a -> a
id x = x

const :: a -> b -> a
const x _ = x

(.) :: (b -> c) -> (a -> b) -> a -> c
(f . g) x = f (g x)
infixr 9 .

flip :: (a -> b -> c) -> b -> a -> c
flip f y x = f x y

($) :: (a -> b) -> a -> b
f $ x = f x
infixr 0 $

(&) :: a -> (a -> b) -> b
x & f = f x
infixl 1 &

on :: (b -> b -> c) -> (a -> b) -> a -> a -> c
on op f x y = f x `op` f y

fix :: (a -> a) -> a
fix f = let x = f x in x

seq :: a -> b -> b
seq = seq  -- Primitive

($!) :: (a -> b) -> a -> b
f $! x = x `seq` f x
infixr 0 $!

-- List operations with fusion
{-# RULES
"map/map"       forall f g xs. map f (map g xs) = map (f . g) xs
"map/coerce"    map coerce = coerce
"fold/build"    forall k z (g :: forall b. (a -> b -> b) -> b -> b). foldr k z (build g) = g k z
"foldr/augment" forall k z xs (g :: forall b. (a -> b -> b) -> b -> b). foldr k z (augment g xs) = g k (foldr k z xs)
"foldr/id"      foldr (:) [] = id
"foldr/app"     forall ys. foldr (:) ys = (++ ys)
"++"            forall xs ys. xs ++ ys = augment (\c n -> foldr c n xs) ys
  #-}

map :: (a -> b) -> [a] -> [b]
map _ []     = []
map f (x:xs) = f x : map f xs
{-# NOINLINE [1] map #-}

(++) :: [a] -> [a] -> [a]
[]     ++ ys = ys
(x:xs) ++ ys = x : (xs ++ ys)
{-# NOINLINE [1] (++) #-}

filter :: (a -> Bool) -> [a] -> [a]
filter _ []     = []
filter p (x:xs)
    | p x       = x : filter p xs
    | otherwise = filter p xs
{-# NOINLINE [1] filter #-}

head :: [a] -> a
head (x:_) = x
head []    = error "head: empty list"

last :: [a] -> a
last [x]    = x
last (_:xs) = last xs
last []     = error "last: empty list"

tail :: [a] -> [a]
tail (_:xs) = xs
tail []     = error "tail: empty list"

init :: [a] -> [a]
init [_]    = []
init (x:xs) = x : init xs
init []     = error "init: empty list"

null :: [a] -> Bool
null []    = True
null (_:_) = False

length :: [a] -> Int
length = foldl' (\c _ -> c + 1) 0

(!!) :: [a] -> Int -> a
xs !! n | n < 0 = error "!!: negative index"
[]     !! _     = error "!!: index too large"
(x:_)  !! 0     = x
(_:xs) !! n     = xs !! (n - 1)

reverse :: [a] -> [a]
reverse = foldl (flip (:)) []

take :: Int -> [a] -> [a]
take n _ | n <= 0 = []
take _ []         = []
take n (x:xs)     = x : take (n - 1) xs

drop :: Int -> [a] -> [a]
drop n xs | n <= 0 = xs
drop _ []          = []
drop n (_:xs)      = drop (n - 1) xs

splitAt :: Int -> [a] -> ([a], [a])
splitAt n xs = (take n xs, drop n xs)

takeWhile :: (a -> Bool) -> [a] -> [a]
takeWhile _ []     = []
takeWhile p (x:xs)
    | p x       = x : takeWhile p xs
    | otherwise = []

dropWhile :: (a -> Bool) -> [a] -> [a]
dropWhile _ []     = []
dropWhile p xs@(x:xs')
    | p x       = dropWhile p xs'
    | otherwise = xs

span :: (a -> Bool) -> [a] -> ([a], [a])
span _ []     = ([], [])
span p xs@(x:xs')
    | p x       = let (ys, zs) = span p xs' in (x:ys, zs)
    | otherwise = ([], xs)

break :: (a -> Bool) -> [a] -> ([a], [a])
break p = span (not . p)

elem :: Eq a => a -> [a] -> Bool
elem _ []     = False
elem x (y:ys) = x == y || elem x ys

notElem :: Eq a => a -> [a] -> Bool
notElem x = not . elem x

lookup :: Eq a => a -> [(a, b)] -> Maybe b
lookup _ []          = Nothing
lookup k ((x, v):xs)
    | k == x         = Just v
    | otherwise      = lookup k xs

find :: (a -> Bool) -> [a] -> Maybe a
find _ []     = Nothing
find p (x:xs)
    | p x       = Just x
    | otherwise = find p xs

zip :: [a] -> [b] -> [(a, b)]
zip = zipWith (,)

zip3 :: [a] -> [b] -> [c] -> [(a, b, c)]
zip3 = zipWith3 (,,)

zipWith :: (a -> b -> c) -> [a] -> [b] -> [c]
zipWith _ []     _      = []
zipWith _ _      []     = []
zipWith f (x:xs) (y:ys) = f x y : zipWith f xs ys

zipWith3 :: (a -> b -> c -> d) -> [a] -> [b] -> [c] -> [d]
zipWith3 _ []     _      _      = []
zipWith3 _ _      []     _      = []
zipWith3 _ _      _      []     = []
zipWith3 f (x:xs) (y:ys) (z:zs) = f x y z : zipWith3 f xs ys zs

unzip :: [(a, b)] -> ([a], [b])
unzip = foldr (\(a, b) (as, bs) -> (a:as, b:bs)) ([], [])

unzip3 :: [(a, b, c)] -> ([a], [b], [c])
unzip3 = foldr (\(a, b, c) (as, bs, cs) -> (a:as, b:bs, c:cs)) ([], [], [])

iterate :: (a -> a) -> a -> [a]
iterate f x = x : iterate f (f x)

repeat :: a -> [a]
repeat x = xs where xs = x : xs

replicate :: Int -> a -> [a]
replicate n x = take n (repeat x)

cycle :: [a] -> [a]
cycle [] = error "cycle: empty list"
cycle xs = xs' where xs' = xs ++ xs'

foldr1 :: (a -> a -> a) -> [a] -> a
foldr1 _ [x]    = x
foldr1 f (x:xs) = f x (foldr1 f xs)
foldr1 _ []     = error "foldr1: empty list"

foldl1 :: (a -> a -> a) -> [a] -> a
foldl1 f (x:xs) = foldl f x xs
foldl1 _ []     = error "foldl1: empty list"

and :: [Bool] -> Bool
and = foldr (&&) True

or :: [Bool] -> Bool
or = foldr (||) False

any :: (a -> Bool) -> [a] -> Bool
any p = or . map p

all :: (a -> Bool) -> [a] -> Bool
all p = and . map p

sum :: Num a => [a] -> a
sum = foldl' (+) 0

product :: Num a => [a] -> a
product = foldl' (*) 1

concat :: [[a]] -> [a]
concat = foldr (++) []

concatMap :: (a -> [b]) -> [a] -> [b]
concatMap f = concat . map f

maximum :: Ord a => [a] -> a
maximum = foldl1 max

minimum :: Ord a => [a] -> a
minimum = foldl1 min

scanl :: (b -> a -> b) -> b -> [a] -> [b]
scanl f q ls = q : case ls of
    []   -> []
    x:xs -> scanl f (f q x) xs

scanr :: (a -> b -> b) -> b -> [a] -> [b]
scanr _ q0 []     = [q0]
scanr f q0 (x:xs) = f x q : qs
  where qs@(q:_) = scanr f q0 xs

lines :: String -> [String]
lines ""   = []
lines s    = let (l, s') = break (== '\n') s
             in l : case s' of
                      []      -> []
                      (_:s'') -> lines s''

words :: String -> [String]
words s = case dropWhile isSpace s of
    "" -> []
    s' -> w : words s''
      where (w, s'') = break isSpace s'
  where isSpace c = c `elem` " \t\n\r"

unlines :: [String] -> String
unlines = concatMap (++ "\n")

unwords :: [String] -> String
unwords []     = ""
unwords [w]    = w
unwords (w:ws) = w ++ ' ' : unwords ws

sort :: Ord a => [a] -> [a]
sort = sortBy compare

sortBy :: (a -> a -> Ordering) -> [a] -> [a]
sortBy cmp = mergeAll . sequences
  where
    sequences (a:b:xs)
      | cmp a b == GT = descending b [a] xs
      | otherwise     = ascending b (a:) xs
    sequences xs      = [xs]
    
    descending a as (b:bs)
      | cmp a b == GT = descending b (a:as) bs
    descending a as bs    = (a:as) : sequences bs
    
    ascending a as (b:bs)
      | cmp a b /= GT = ascending b (\ys -> as (a:ys)) bs
    ascending a as bs     = let !x = as [a] in x : sequences bs
    
    mergeAll [x] = x
    mergeAll xs  = mergeAll (mergePairs xs)
    
    mergePairs (a:b:xs) = let !x = merge a b in x : mergePairs xs
    mergePairs xs       = xs
    
    merge as@(a:as') bs@(b:bs')
      | cmp a b == GT = b : merge as bs'
      | otherwise     = a : merge as' bs
    merge [] bs       = bs
    merge as []       = as

sortOn :: Ord b => (a -> b) -> [a] -> [a]
sortOn f = map snd . sortBy (compare `on` fst) . map (\x -> (f x, x))

nub :: Eq a => [a] -> [a]
nub = nubBy (==)

nubBy :: (a -> a -> Bool) -> [a] -> [a]
nubBy eq = go []
  where go _ []     = []
        go seen (x:xs)
            | any (eq x) seen = go seen xs
            | otherwise       = x : go (x:seen) xs

delete :: Eq a => a -> [a] -> [a]
delete = deleteBy (==)

deleteBy :: (a -> a -> Bool) -> a -> [a] -> [a]
deleteBy _  _ []     = []
deleteBy eq x (y:ys)
    | x `eq` y       = ys
    | otherwise      = y : deleteBy eq x ys

(\\) :: Eq a => [a] -> [a] -> [a]
(\\) = foldl (flip delete)
infixl 9 \\

union :: Eq a => [a] -> [a] -> [a]
union xs ys = xs ++ foldl (flip delete) (nub ys) xs

intersect :: Eq a => [a] -> [a] -> [a]
intersect xs ys = [x | x <- xs, x `elem` ys]

-- Numeric operations
subtract :: Num a => a -> a -> a
subtract x y = y - x

even :: Integral a => a -> Bool
even n = n `rem` 2 == 0

odd :: Integral a => a -> Bool
odd = not . even

gcd :: Integral a => a -> a -> a
gcd x y = gcd' (abs x) (abs y)
  where gcd' a 0 = a
        gcd' a b = gcd' b (a `rem` b)

lcm :: Integral a => a -> a -> a
lcm _ 0 = 0
lcm 0 _ = 0
lcm x y = abs ((x `quot` gcd x y) * y)

(^) :: (Num a, Integral b) => a -> b -> a
x ^ 0         = 1
x ^ n | n > 0 = x * (x ^ (n - 1))
_ ^ _         = error "^: negative exponent"
infixr 8 ^

(^^) :: (Fractional a, Integral b) => a -> b -> a
x ^^ n = if n >= 0 then x ^ n else recip (x ^ negate n)
infixr 8 ^^

fromIntegral :: (Integral a, Num b) => a -> b
fromIntegral = fromInteger . toInteger

realToFrac :: (Real a, Fractional b) => a -> b
realToFrac = fromRational . toRational

-- Monad utilities
mapM :: (Traversable t, Monad m) => (a -> m b) -> t a -> m (t b)
mapM = traverse

mapM_ :: (Foldable t, Monad m) => (a -> m b) -> t a -> m ()
mapM_ f = foldr ((>>) . f) (return ())

forM :: (Traversable t, Monad m) => t a -> (a -> m b) -> m (t b)
forM = flip mapM

forM_ :: (Foldable t, Monad m) => t a -> (a -> m b) -> m ()
forM_ = flip mapM_

sequence_ :: (Foldable t, Monad m) => t (m a) -> m ()
sequence_ = foldr (>>) (return ())

when :: Applicative f => Bool -> f () -> f ()
when p s = if p then s else pure ()

unless :: Applicative f => Bool -> f () -> f ()
unless p s = if p then pure () else s

guard :: Alternative f => Bool -> f ()
guard True  = pure ()
guard False = empty

filterM :: Applicative m => (a -> m Bool) -> [a] -> m [a]
filterM _ []     = pure []
filterM p (x:xs) = liftA2 (\b ys -> if b then x:ys else ys) (p x) (filterM p xs)

foldM :: (Foldable t, Monad m) => (b -> a -> m b) -> b -> t a -> m b
foldM f z0 xs = foldr c return xs z0
  where c x k z = f z x >>= k

foldM_ :: (Foldable t, Monad m) => (b -> a -> m b) -> b -> t a -> m ()
foldM_ f z xs = foldM f z xs >> return ()

replicateM :: Applicative m => Int -> m a -> m [a]
replicateM n x = sequenceA (replicate n x)

replicateM_ :: Applicative m => Int -> m a -> m ()
replicateM_ n x = sequence_ (replicate n x)

-- Error handling
error :: String -> a
error s = errorWithoutStackTrace s

errorWithoutStackTrace :: String -> a
errorWithoutStackTrace = error

undefined :: a
undefined = error "undefined"

fail :: MonadFail m => String -> m a
fail = Prelude.fail

class Monad m => MonadFail m where
    fail :: String -> m a

-- IO type and operations (primitives from runtime)
data IO a  -- Abstract

instance Functor IO where
    fmap f x = x >>= (return . f)

instance Applicative IO where
    pure = return
    (<*>) = ap

instance Monad IO where
    return = returnIO
    (>>=)  = bindIO

-- IO primitives (implemented in runtime)
foreign import ccall "bhc_putChar" putChar :: Char -> IO ()
foreign import ccall "bhc_putStr" putStr :: String -> IO ()

putStrLn :: String -> IO ()
putStrLn s = putStr s >> putChar '\n'

print :: Show a => a -> IO ()
print = putStrLn . show

foreign import ccall "bhc_getChar" getChar :: IO Char
foreign import ccall "bhc_getLine" getLine :: IO String
foreign import ccall "bhc_getContents" getContents :: IO String

foreign import ccall "bhc_readFile" readFile :: FilePath -> IO String
foreign import ccall "bhc_writeFile" writeFile :: FilePath -> String -> IO ()
foreign import ccall "bhc_appendFile" appendFile :: FilePath -> String -> IO ()

interact :: (String -> String) -> IO ()
interact f = getContents >>= putStr . f

-- Primitive bindings
foreign import ccall "bhc_returnIO" returnIO :: a -> IO a
foreign import ccall "bhc_bindIO" bindIO :: IO a -> (a -> IO b) -> IO b

-- Type aliases
type String = [Char]
type FilePath = String

-- Build/augment for fusion
build :: (forall b. (a -> b -> b) -> b -> b) -> [a]
build g = g (:) []
{-# INLINE [1] build #-}

augment :: (forall b. (a -> b -> b) -> b -> b) -> [a] -> [a]
augment g xs = g (:) xs
{-# INLINE [1] augment #-}

-- Coercion helper
coerce :: a -> b
coerce = error "coerce"
