-- |
-- Module      : BHC.Control.Monad
-- Description : Monad class and operations
-- Copyright   : (c) BHC Contributors, 2026
-- License     : BSD-3-Clause
-- Stability   : stable

module BHC.Control.Monad (
    -- * Functor and Monad classes
    Functor(..),
    Applicative(..),
    Monad(..),
    MonadFail(..),
    
    -- * Functions
    -- ** Naming conventions
    -- | Functions with @M@ suffix work in a monad.
    -- Functions with @_@ suffix discard the result.
    
    -- ** Basic functions
    (=<<),
    (>=>), (<=<),
    forever,
    void,
    
    -- ** Generalisations of list functions
    join,
    msum, mfilter,
    filterM, mapAndUnzipM,
    zipWithM, zipWithM_,
    foldM, foldM_,
    replicateM, replicateM_,
    
    -- ** Conditional execution
    guard,
    when, unless,
    
    -- ** Monadic lifting
    liftM, liftM2, liftM3, liftM4, liftM5,
    ap,
    (<$!>),
    
    -- * MonadPlus
    MonadPlus(..),
) where

import BHC.Prelude

-- ------------------------------------------------------------
-- Monadic lifting
-- ------------------------------------------------------------

-- | Strict version of '<$>'. Forces the result of the function
-- application before wrapping in the monad.
--
-- >>> Just 1 <$!> (+1)
-- Just 2
(<$!>) :: Monad m => (a -> b) -> m a -> m b
f <$!> m = m >>= \x -> let z = f x in z `seq` return z
infixl 4 <$!>

-- | Lift a function to a monad.
-- Equivalent to 'fmap' but works with the 'Monad' constraint.
--
-- >>> liftM (+1) (Just 2)
-- Just 3
-- >>> liftM (+1) [1, 2, 3]
-- [2,3,4]
liftM :: Monad m => (a -> b) -> m a -> m b
liftM f m = m >>= return . f

-- | Lift a binary function to a monad.
--
-- >>> liftM2 (+) (Just 1) (Just 2)
-- Just 3
-- >>> liftM2 (+) [1, 2] [10, 20]
-- [11,21,12,22]
liftM2 :: Monad m => (a -> b -> c) -> m a -> m b -> m c
liftM2 f m1 m2 = do
    x1 <- m1
    x2 <- m2
    return (f x1 x2)

-- | Lift a ternary function to a monad.
liftM3 :: Monad m => (a -> b -> c -> d) -> m a -> m b -> m c -> m d
liftM3 f m1 m2 m3 = do
    x1 <- m1
    x2 <- m2
    x3 <- m3
    return (f x1 x2 x3)

-- | Lift a quaternary function to a monad.
liftM4 :: Monad m => (a -> b -> c -> d -> e) -> m a -> m b -> m c -> m d -> m e
liftM4 f m1 m2 m3 m4 = do
    x1 <- m1
    x2 <- m2
    x3 <- m3
    x4 <- m4
    return (f x1 x2 x3 x4)

-- | Lift a quinary function to a monad.
liftM5 :: Monad m => (a -> b -> c -> d -> e -> f) -> m a -> m b -> m c -> m d -> m e -> m f
liftM5 f m1 m2 m3 m4 m5 = do
    x1 <- m1
    x2 <- m2
    x3 <- m3
    x4 <- m4
    x5 <- m5
    return (f x1 x2 x3 x4 x5)

-- ------------------------------------------------------------
-- MonadPlus operations
-- ------------------------------------------------------------

-- | Combine a foldable of 'MonadPlus' values using 'mplus'.
--
-- >>> msum [Nothing, Just 1, Just 2]
-- Just 1
-- >>> msum [[], [1, 2], [3]]
-- [1,2,3]
msum :: (Foldable t, MonadPlus m) => t (m a) -> m a
msum = foldr mplus mzero

-- | Filter values that satisfy a predicate in a 'MonadPlus'.
--
-- >>> mfilter even (Just 2)
-- Just 2
-- >>> mfilter even (Just 3)
-- Nothing
mfilter :: MonadPlus m => (a -> Bool) -> m a -> m a
mfilter p ma = do
    a <- ma
    if p a then return a else mzero
