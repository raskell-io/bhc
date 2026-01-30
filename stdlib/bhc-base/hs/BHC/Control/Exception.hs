-- |
-- Module      : BHC.Control.Exception
-- Description : Exception handling primitives
-- Copyright   : (c) BHC Contributors, 2026
-- License     : BSD-3-Clause
-- Stability   : stable
--
-- Unified exception hierarchy and combinators for BHC.
-- This module provides the core exception mechanism built on
-- the RTS throw/catch primitives.

module BHC.Control.Exception (
    -- * The Exception type
    SomeException(..),
    Exception(..),

    -- * Common exception types
    IOException(..),
    ErrorCall(..),
    ArithException(..),
    AssertionFailed(..),
    AsyncException(..),

    -- * Throwing exceptions
    throw,
    throwIO,

    -- * Catching exceptions
    catch,
    catchJust,
    handle,
    handleJust,
    try,
    tryJust,

    -- * Exception transformations
    mapException,

    -- * Resource management
    bracket,
    bracket_,
    bracketOnError,
    finally,
    onException,

    -- * Evaluation
    evaluate,

    -- * Async exception stubs
    mask,
    mask_,
    uninterruptibleMask,
    uninterruptibleMask_,
    getMaskingState,
    MaskingState(..),
    throwTo,
) where

import BHC.Prelude hiding (catch)

-- ============================================================
-- The Exception type hierarchy
-- ============================================================

-- | The root exception type. All exceptions can be converted
-- to and from 'SomeException'.
data SomeException = SomeException String
  deriving (Show, Eq)

-- | The 'Exception' typeclass. Types that can be thrown and
-- caught as exceptions.
class (Show e) => Exception e where
  -- | Convert to the root exception type.
  toException :: e -> SomeException
  toException e = SomeException (show e)

  -- | Try to convert from the root exception type.
  fromException :: SomeException -> Maybe e
  fromException _ = Nothing

  -- | Render a human-readable representation.
  displayException :: e -> String
  displayException = show

instance Exception SomeException where
  toException = id
  fromException = Just

-- ============================================================
-- Common exception types
-- ============================================================

-- | An exception from the I/O subsystem.
data IOException = IOException String
  deriving (Show, Eq)

instance Exception IOException

-- | An exception from calling 'error'.
data ErrorCall = ErrorCall String
  deriving (Show, Eq)

instance Exception ErrorCall

-- | Arithmetic exceptions (division by zero, overflow, etc.).
data ArithException
  = Overflow
  | Underflow
  | LossOfPrecision
  | DivideByZero
  | Denormal
  | RatioZeroDenominator
  deriving (Show, Eq, Ord, Enum, Bounded)

instance Exception ArithException

-- | An assertion failure.
data AssertionFailed = AssertionFailed String
  deriving (Show, Eq)

instance Exception AssertionFailed

-- | Asynchronous exceptions.
data AsyncException
  = StackOverflow
  | HeapOverflow
  | ThreadKilled
  | UserInterrupt
  deriving (Show, Eq, Ord, Enum, Bounded)

instance Exception AsyncException

-- | Masking state for asynchronous exceptions.
data MaskingState
  = Unmasked
  | MaskedInterruptible
  | MaskedUninterruptible
  deriving (Show, Eq, Ord)

-- ============================================================
-- FFI imports
-- ============================================================

foreign import ccall "bhc_throw"
  primThrow :: SomeException -> a

foreign import ccall "bhc_catch"
  primCatch :: IO a -> (SomeException -> IO a) -> IO a

foreign import ccall "bhc_evaluate"
  primEvaluate :: a -> IO a

-- ============================================================
-- Throwing exceptions
-- ============================================================

-- | Throw an exception. Can be called from pure code.
throw :: Exception e => e -> a
throw = primThrow . toException

-- | Throw an exception in the IO monad.
throwIO :: Exception e => e -> IO a
throwIO e = do
  _ <- return ()
  throw e

-- ============================================================
-- Catching exceptions
-- ============================================================

-- | Catch an exception of a particular type.
catch :: Exception e => IO a -> (e -> IO a) -> IO a
catch action handler = primCatch action handler'
  where
    handler' se = case fromException se of
      Just e  -> handler e
      Nothing -> primThrow se

-- | Catch an exception with a selector function.
catchJust :: Exception e
          => (e -> Maybe b)
          -> IO a
          -> (b -> IO a)
          -> IO a
catchJust p action handler = catch action handler'
  where
    handler' e = case p e of
      Just b  -> handler b
      Nothing -> throwIO e

-- | A version of 'catch' with the arguments swapped.
handle :: Exception e => (e -> IO a) -> IO a -> IO a
handle = flip catch

-- | A version of 'catchJust' with the arguments swapped.
handleJust :: Exception e
           => (e -> Maybe b)
           -> (b -> IO a)
           -> IO a
           -> IO a
handleJust p handler action = catchJust p action handler

-- | Try running an action, returning either the exception or the result.
try :: Exception e => IO a -> IO (Either e a)
try action = catch (fmap Right action) (return . Left)

-- | A version of 'try' with a selector function.
tryJust :: Exception e
        => (e -> Maybe b)
        -> IO a
        -> IO (Either b a)
tryJust p action = do
  r <- try action
  case r of
    Right v -> return (Right v)
    Left e  -> case p e of
      Just b  -> return (Left b)
      Nothing -> throwIO e

-- ============================================================
-- Exception transformations
-- ============================================================

-- | Map one exception type to another.
mapException :: (Exception e1, Exception e2)
             => (e1 -> e2) -> a -> a
mapException f v = unsafePerformIO $ catch (evaluate v) (\e -> throwIO (f e))
  where
    {-# NOINLINE unsafePerformIO #-}
    unsafePerformIO :: IO a -> a
    unsafePerformIO = error "unsafePerformIO: implemented via compiler magic"

-- ============================================================
-- Resource management
-- ============================================================

-- | Acquire a resource, perform an action, then release
-- the resource. The resource is released even if an
-- exception is raised.
--
-- @
-- bracket (openFile f m) hClose $ \\h -> do
--     contents <- hGetContents h
--     process contents
-- @
bracket :: IO a        -- ^ Acquire resource
        -> (a -> IO b) -- ^ Release resource
        -> (a -> IO c) -- ^ Action with resource
        -> IO c
bracket acquire release action = do
  resource <- acquire
  result <- action resource `onException` release resource
  _ <- release resource
  return result

-- | Like 'bracket', but discards the resource.
bracket_ :: IO a -> IO b -> IO c -> IO c
bracket_ acquire release action =
  bracket acquire (const release) (const action)

-- | Like 'bracket', but only releases the resource on exception.
bracketOnError :: IO a -> (a -> IO b) -> (a -> IO c) -> IO c
bracketOnError acquire release action = do
  resource <- acquire
  action resource `onException` release resource

-- | Perform an action then a cleanup, even if an exception
-- is raised.
--
-- @
-- finally action cleanup
-- @
finally :: IO a -> IO b -> IO a
finally action cleanup = do
  result <- action `onException` cleanup
  _ <- cleanup
  return result

-- | Perform an action, and if an exception is raised, run
-- the cleanup before re-raising.
onException :: IO a -> IO b -> IO a
onException action cleanup =
  action `catch` \e -> do
    _ <- cleanup
    throw (e :: SomeException)

-- ============================================================
-- Evaluation
-- ============================================================

-- | Force a value to Weak Head Normal Form.
--
-- This is useful for catching exceptions that might be lurking
-- inside unevaluated thunks.
--
-- @
-- evaluate (error "boom")  -- raises the exception immediately
-- @
evaluate :: a -> IO a
evaluate = primEvaluate

-- ============================================================
-- Async exception stubs
-- ============================================================

-- | Mask asynchronous exceptions during the execution of an action.
--
-- Currently a stub — just runs the action, since BHC does not
-- yet support asynchronous exceptions.
mask :: ((IO a -> IO a) -> IO b) -> IO b
mask action = action id

-- | Like 'mask', but does not pass a restore action.
mask_ :: IO a -> IO a
mask_ action = mask (\_ -> action)

-- | Like 'mask', but cannot be interrupted.
--
-- Currently a stub — same as 'mask'.
uninterruptibleMask :: ((IO a -> IO a) -> IO b) -> IO b
uninterruptibleMask action = action id

-- | Like 'uninterruptibleMask', but does not pass a restore action.
uninterruptibleMask_ :: IO a -> IO a
uninterruptibleMask_ action = uninterruptibleMask (\_ -> action)

-- | Get the current masking state.
--
-- Currently always returns 'Unmasked'.
getMaskingState :: IO MaskingState
getMaskingState = return Unmasked

-- | Throw an exception to another thread.
--
-- Currently a stub — throws locally.
throwTo :: Exception e => Int -> e -> IO ()
throwTo _ e = throwIO e
