-- Test: transformers_props
-- Category: semantic
-- Profile: default
-- Expected: success
-- Spec: H26-SPEC Phase 5 - Monad Transformer Laws

{-# HASKELL_EDITION 2026 #-}

module TransformersPropsTest where

import BHC.Prelude
import BHC.Control.Monad.Trans
import BHC.Control.Monad.Identity
import BHC.Control.Monad.Reader
import BHC.Control.Monad.Writer
import BHC.Control.Monad.State
import BHC.Control.Monad.Except
import BHC.Control.Monad.Maybe
import BHC.Control.Monad.RWS
import BHC.Control.Monad.Cont

-- ================================================================
-- Test Infrastructure
-- ================================================================

-- Simple test values for property testing
testInts :: [Int]
testInts = [-100, -1, 0, 1, 42, 100, 999]

testStrings :: [String]
testStrings = ["", "a", "hello", "world", "test string"]

-- Check if a property holds for all test values
allHold :: [Bool] -> Bool
allHold = and

-- ================================================================
-- Functor Laws
-- ================================================================

-- Law 1: fmap id = id
-- Law 2: fmap (f . g) = fmap f . fmap g

-- Identity Functor Laws
propIdentityFunctorIdentity :: Bool
propIdentityFunctorIdentity =
    allHold [fmap id (Identity x) == Identity x | x <- testInts]
-- Result: True

propIdentityFunctorComposition :: Bool
propIdentityFunctorComposition =
    let f = (+1)
        g = (*2)
    in allHold [fmap (f . g) (Identity x) == (fmap f . fmap g) (Identity x) | x <- testInts]
-- Result: True

-- ReaderT Functor Laws
propReaderTFunctorIdentity :: Bool
propReaderTFunctorIdentity =
    allHold [runReader (fmap id (reader id)) r == runReader (reader id) r | r <- testInts]
-- Result: True

propReaderTFunctorComposition :: Bool
propReaderTFunctorComposition =
    let f = (+1)
        g = (*2)
        m = reader id
    in allHold [runReader (fmap (f . g) m) r == runReader ((fmap f . fmap g) m) r | r <- testInts]
-- Result: True

-- WriterT Functor Laws
propWriterTFunctorIdentity :: Bool
propWriterTFunctorIdentity =
    allHold [runWriter (fmap id (writer (x, "log"))) == runWriter (writer (x, "log")) | x <- testInts]
-- Result: True

propWriterTFunctorComposition :: Bool
propWriterTFunctorComposition =
    let f = (+1)
        g = (*2)
    in allHold [runWriter (fmap (f . g) (writer (x, "log"))) ==
                runWriter ((fmap f . fmap g) (writer (x, "log"))) | x <- testInts]
-- Result: True

-- StateT Functor Laws
propStateTFunctorIdentity :: Bool
propStateTFunctorIdentity =
    allHold [runState (fmap id (state (\s -> (s, s+1)))) s0 ==
             runState (state (\s -> (s, s+1))) s0 | s0 <- testInts]
-- Result: True

propStateTFunctorComposition :: Bool
propStateTFunctorComposition =
    let f = (+1)
        g = (*2)
        m = state (\s -> (s, s+1))
    in allHold [runState (fmap (f . g) m) s0 == runState ((fmap f . fmap g) m) s0 | s0 <- testInts]
-- Result: True

-- ExceptT Functor Laws
propExceptTFunctorIdentity :: Bool
propExceptTFunctorIdentity =
    allHold [runExcept (fmap id (return x :: Except String Int)) ==
             runExcept (return x :: Except String Int) | x <- testInts]
-- Result: True

propExceptTFunctorComposition :: Bool
propExceptTFunctorComposition =
    let f = (+1)
        g = (*2)
    in allHold [runExcept (fmap (f . g) (return x :: Except String Int)) ==
                runExcept ((fmap f . fmap g) (return x :: Except String Int)) | x <- testInts]
-- Result: True

-- MaybeT Functor Laws
propMaybeTFunctorIdentity :: Bool
propMaybeTFunctorIdentity =
    allHold [runIdentity (runMaybeT (fmap id (return x))) ==
             runIdentity (runMaybeT (return x)) | x <- testInts]
-- Result: True

propMaybeTFunctorComposition :: Bool
propMaybeTFunctorComposition =
    let f = (+1)
        g = (*2)
    in allHold [runIdentity (runMaybeT (fmap (f . g) (return x))) ==
                runIdentity (runMaybeT ((fmap f . fmap g) (return x))) | x <- testInts]
-- Result: True

-- ContT Functor Laws
propContTFunctorIdentity :: Bool
propContTFunctorIdentity =
    allHold [evalCont (fmap id (return x)) == evalCont (return x) | x <- testInts]
-- Result: True

propContTFunctorComposition :: Bool
propContTFunctorComposition =
    let f = (+1)
        g = (*2)
    in allHold [evalCont (fmap (f . g) (return x)) == evalCont ((fmap f . fmap g) (return x)) | x <- testInts]
-- Result: True

-- ================================================================
-- Applicative Laws
-- ================================================================

-- Law 1: pure id <*> v = v                              (Identity)
-- Law 2: pure f <*> pure x = pure (f x)                 (Homomorphism)
-- Law 3: u <*> pure y = pure ($ y) <*> u                (Interchange)
-- Law 4: pure (.) <*> u <*> v <*> w = u <*> (v <*> w)   (Composition)

-- Identity Applicative Laws
propIdentityApplicativeIdentity :: Bool
propIdentityApplicativeIdentity =
    allHold [(pure id <*> Identity x) == Identity x | x <- testInts]
-- Result: True

propIdentityApplicativeHomomorphism :: Bool
propIdentityApplicativeHomomorphism =
    let f = (+1)
    in allHold [(pure f <*> pure x) == (pure (f x) :: Identity Int) | x <- testInts]
-- Result: True

propIdentityApplicativeInterchange :: Bool
propIdentityApplicativeInterchange =
    let u = Identity (+1)
    in allHold [(u <*> pure y) == (pure ($ y) <*> u) | y <- testInts]
-- Result: True

propIdentityApplicativeComposition :: Bool
propIdentityApplicativeComposition =
    let u = Identity (+1)
        v = Identity (*2)
        w = Identity 5
    in (pure (.) <*> u <*> v <*> w) == (u <*> (v <*> w))
-- Result: True

-- ReaderT Applicative Laws
propReaderTApplicativeIdentity :: Bool
propReaderTApplicativeIdentity =
    allHold [runReader (pure id <*> reader (+1)) r == runReader (reader (+1)) r | r <- testInts]
-- Result: True

propReaderTApplicativeHomomorphism :: Bool
propReaderTApplicativeHomomorphism =
    let f = (+1)
    in allHold [runReader (pure f <*> pure x) 0 == runReader (pure (f x) :: Reader Int Int) 0 | x <- testInts]
-- Result: True

-- StateT Applicative Laws
propStateTApplicativeIdentity :: Bool
propStateTApplicativeIdentity =
    let v = state (\s -> (s * 2, s + 1))
    in allHold [runState (pure id <*> v) s == runState v s | s <- testInts]
-- Result: True

propStateTApplicativeHomomorphism :: Bool
propStateTApplicativeHomomorphism =
    let f = (+1)
    in allHold [runState (pure f <*> pure x) s == runState (pure (f x) :: State Int Int) s | x <- testInts, s <- [0, 1]]
-- Result: True

-- ExceptT Applicative Laws
propExceptTApplicativeIdentity :: Bool
propExceptTApplicativeIdentity =
    allHold [runExcept (pure id <*> (return x :: Except String Int)) ==
             runExcept (return x :: Except String Int) | x <- testInts]
-- Result: True

propExceptTApplicativeHomomorphism :: Bool
propExceptTApplicativeHomomorphism =
    let f = (+1)
    in allHold [runExcept (pure f <*> pure x :: Except String Int) ==
                runExcept (pure (f x) :: Except String Int) | x <- testInts]
-- Result: True

-- ================================================================
-- Monad Laws
-- ================================================================

-- Law 1: return a >>= k  =  k a                         (Left Identity)
-- Law 2: m >>= return    =  m                           (Right Identity)
-- Law 3: m >>= (\x -> k x >>= h)  =  (m >>= k) >>= h    (Associativity)

-- Identity Monad Laws
propIdentityMonadLeftIdentity :: Bool
propIdentityMonadLeftIdentity =
    let k = \x -> Identity (x + 1)
    in allHold [(return a >>= k) == k a | a <- testInts]
-- Result: True

propIdentityMonadRightIdentity :: Bool
propIdentityMonadRightIdentity =
    allHold [(Identity m >>= return) == Identity m | m <- testInts]
-- Result: True

propIdentityMonadAssociativity :: Bool
propIdentityMonadAssociativity =
    let k = \x -> Identity (x + 1)
        h = \x -> Identity (x * 2)
        m = Identity 5
    in (m >>= (\x -> k x >>= h)) == ((m >>= k) >>= h)
-- Result: True

-- ReaderT Monad Laws
propReaderTMonadLeftIdentity :: Bool
propReaderTMonadLeftIdentity =
    let k = \x -> reader (\r -> x + r)
    in allHold [runReader (return a >>= k) r == runReader (k a) r | a <- testInts, r <- [0, 1, 10]]
-- Result: True

propReaderTMonadRightIdentity :: Bool
propReaderTMonadRightIdentity =
    let m = reader (*2)
    in allHold [runReader (m >>= return) r == runReader m r | r <- testInts]
-- Result: True

propReaderTMonadAssociativity :: Bool
propReaderTMonadAssociativity =
    let k = \x -> reader (\r -> x + r)
        h = \x -> reader (\r -> x * r)
        m = reader id
    in allHold [runReader (m >>= (\x -> k x >>= h)) r == runReader ((m >>= k) >>= h) r | r <- [1, 2, 5]]
-- Result: True

-- WriterT Monad Laws
propWriterTMonadLeftIdentity :: Bool
propWriterTMonadLeftIdentity =
    let k = \x -> writer (x + 1, "k")
    in allHold [runWriter (return a >>= k) == runWriter (k a) | a <- testInts]
-- Result: True

propWriterTMonadRightIdentity :: Bool
propWriterTMonadRightIdentity =
    allHold [runWriter (writer (m, "log") >>= return) == runWriter (writer (m, "log")) | m <- testInts]
-- Result: True

propWriterTMonadAssociativity :: Bool
propWriterTMonadAssociativity =
    let k = \x -> writer (x + 1, "k")
        h = \x -> writer (x * 2, "h")
        m = writer (5, "m")
    in runWriter (m >>= (\x -> k x >>= h)) == runWriter ((m >>= k) >>= h)
-- Result: True

-- StateT Monad Laws
propStateTMonadLeftIdentity :: Bool
propStateTMonadLeftIdentity =
    let k = \x -> state (\s -> (x + s, s + 1))
    in allHold [runState (return a >>= k) s == runState (k a) s | a <- testInts, s <- [0, 1]]
-- Result: True

propStateTMonadRightIdentity :: Bool
propStateTMonadRightIdentity =
    let m = state (\s -> (s * 2, s + 1))
    in allHold [runState (m >>= return) s == runState m s | s <- testInts]
-- Result: True

propStateTMonadAssociativity :: Bool
propStateTMonadAssociativity =
    let k = \x -> state (\s -> (x + s, s + 1))
        h = \x -> state (\s -> (x * s, s * 2))
        m = state (\s -> (s, s + 1))
    in allHold [runState (m >>= (\x -> k x >>= h)) s == runState ((m >>= k) >>= h) s | s <- [1, 2, 3]]
-- Result: True

-- ExceptT Monad Laws
propExceptTMonadLeftIdentity :: Bool
propExceptTMonadLeftIdentity =
    let k = \x -> return (x + 1) :: Except String Int
    in allHold [runExcept (return a >>= k) == runExcept (k a) | a <- testInts]
-- Result: True

propExceptTMonadRightIdentity :: Bool
propExceptTMonadRightIdentity =
    allHold [runExcept ((return m :: Except String Int) >>= return) ==
             runExcept (return m :: Except String Int) | m <- testInts]
-- Result: True

propExceptTMonadAssociativity :: Bool
propExceptTMonadAssociativity =
    let k = \x -> return (x + 1) :: Except String Int
        h = \x -> return (x * 2) :: Except String Int
        m = return 5 :: Except String Int
    in runExcept (m >>= (\x -> k x >>= h)) == runExcept ((m >>= k) >>= h)
-- Result: True

-- MaybeT Monad Laws
propMaybeTMonadLeftIdentity :: Bool
propMaybeTMonadLeftIdentity =
    let k = \x -> return (x + 1) :: MaybeT Identity Int
    in allHold [runIdentity (runMaybeT (return a >>= k)) == runIdentity (runMaybeT (k a)) | a <- testInts]
-- Result: True

propMaybeTMonadRightIdentity :: Bool
propMaybeTMonadRightIdentity =
    allHold [runIdentity (runMaybeT ((return m :: MaybeT Identity Int) >>= return)) ==
             runIdentity (runMaybeT (return m :: MaybeT Identity Int)) | m <- testInts]
-- Result: True

propMaybeTMonadAssociativity :: Bool
propMaybeTMonadAssociativity =
    let k = \x -> return (x + 1) :: MaybeT Identity Int
        h = \x -> return (x * 2) :: MaybeT Identity Int
        m = return 5 :: MaybeT Identity Int
    in runIdentity (runMaybeT (m >>= (\x -> k x >>= h))) == runIdentity (runMaybeT ((m >>= k) >>= h))
-- Result: True

-- ContT Monad Laws
propContTMonadLeftIdentity :: Bool
propContTMonadLeftIdentity =
    let k = \x -> return (x + 1) :: Cont Int Int
    in allHold [evalCont (return a >>= k) == evalCont (k a) | a <- testInts]
-- Result: True

propContTMonadRightIdentity :: Bool
propContTMonadRightIdentity =
    allHold [evalCont ((return m :: Cont Int Int) >>= return) == evalCont (return m :: Cont Int Int) | m <- testInts]
-- Result: True

propContTMonadAssociativity :: Bool
propContTMonadAssociativity =
    let k = \x -> return (x + 1) :: Cont Int Int
        h = \x -> return (x * 2) :: Cont Int Int
        m = return 5 :: Cont Int Int
    in evalCont (m >>= (\x -> k x >>= h)) == evalCont ((m >>= k) >>= h)
-- Result: True

-- ================================================================
-- MonadTrans Laws
-- ================================================================

-- Law 1: lift . return = return
-- Law 2: lift (m >>= f) = lift m >>= (lift . f)

propReaderTLiftReturn :: Bool
propReaderTLiftReturn =
    allHold [runReader (lift (Identity x)) r == runReader (return x) r | x <- testInts, r <- [0, 1]]
-- Result: True

propReaderTLiftBind :: Bool
propReaderTLiftBind =
    let m = Identity 5
        f = \x -> Identity (x + 1)
    in allHold [runReader (lift (m >>= f)) r == runReader (lift m >>= (lift . f)) r | r <- testInts]
-- Result: True

propStateTLiftReturn :: Bool
propStateTLiftReturn =
    allHold [runState (lift (Identity x) :: StateT Int Identity Int) s == runState (return x) s | x <- testInts, s <- [0, 1]]
-- Result: True

propStateTLiftBind :: Bool
propStateTLiftBind =
    let m = Identity 5
        f = \x -> Identity (x + 1)
    in allHold [runState (lift (m >>= f) :: StateT Int Identity Int) s ==
                runState (lift m >>= (lift . f)) s | s <- [0, 1]]
-- Result: True

propWriterTLiftReturn :: Bool
propWriterTLiftReturn =
    allHold [runWriter (lift (Identity x) :: WriterT String Identity Int) == runWriter (return x) | x <- testInts]
-- Result: True

propWriterTLiftBind :: Bool
propWriterTLiftBind =
    let m = Identity 5
        f = \x -> Identity (x + 1)
    in runWriter (lift (m >>= f) :: WriterT String Identity Int) == runWriter (lift m >>= (lift . f))
-- Result: True

propExceptTLiftReturn :: Bool
propExceptTLiftReturn =
    allHold [runExceptT (lift (Identity x) :: ExceptT String Identity Int) ==
             runExceptT (return x :: ExceptT String Identity Int) | x <- testInts]
-- Result: True

propExceptTLiftBind :: Bool
propExceptTLiftBind =
    let m = Identity 5
        f = \x -> Identity (x + 1)
    in runExceptT (lift (m >>= f) :: ExceptT String Identity Int) ==
       runExceptT (lift m >>= (lift . f) :: ExceptT String Identity Int)
-- Result: True

propMaybeTLiftReturn :: Bool
propMaybeTLiftReturn =
    allHold [runMaybeT (lift (Identity x)) == runMaybeT (return x :: MaybeT Identity Int) | x <- testInts]
-- Result: True

propMaybeTLiftBind :: Bool
propMaybeTLiftBind =
    let m = Identity 5
        f = \x -> Identity (x + 1)
    in runMaybeT (lift (m >>= f) :: MaybeT Identity Int) == runMaybeT (lift m >>= (lift . f))
-- Result: True

-- ================================================================
-- MTL Class Laws
-- ================================================================

-- MonadReader Laws:
-- ask >>= k = local id (ask >>= k)
-- local f m >>= k = local f (m >>= \a -> local id (k a))

propMonadReaderAskLocal :: Bool
propMonadReaderAskLocal =
    let k = \r -> reader (\_ -> r * 2)
    in allHold [runReader (ask >>= k) r == runReader (local id (ask >>= k)) r | r <- testInts]
-- Result: True

-- MonadWriter Laws:
-- tell mempty = return ()
-- tell (a <> b) = tell a >> tell b

propMonadWriterTellMempty :: Bool
propMonadWriterTellMempty =
    runWriter (tell "" :: Writer String ()) == runWriter (return ())
-- Result: True

propMonadWriterTellAppend :: Bool
propMonadWriterTellAppend =
    allHold [runWriter (tell (a ++ b) :: Writer String ()) ==
             runWriter (tell a >> tell b :: Writer String ()) | a <- testStrings, b <- testStrings]
-- Result: True

-- MonadState Laws:
-- get >>= put = return ()
-- put s >> get = put s >> return s

propMonadStateGetPut :: Bool
propMonadStateGetPut =
    allHold [runState (get >>= put) s == runState (return ()) s | s <- testInts]
-- Result: True

propMonadStatePutGet :: Bool
propMonadStatePutGet =
    allHold [runState (put s' >> get) s == runState (put s' >> return s') s | s <- testInts, s' <- [0, 1, 42]]
-- Result: True

-- put s >> put s' = put s'
propMonadStatePutPut :: Bool
propMonadStatePutPut =
    allHold [runState (put s1 >> put s2) s0 == runState (put s2) s0 | s0 <- [0], s1 <- testInts, s2 <- [1, 2]]
-- Result: True

-- MonadError Laws:
-- throwError e >>= k = throwError e
-- catchError (throwError e) h = h e
-- catchError (return a) h = return a

propMonadErrorThrowBind :: Bool
propMonadErrorThrowBind =
    let k = \x -> return (x + 1) :: Except String Int
    in allHold [runExcept (throwE e >>= k) == runExcept (throwE e :: Except String Int) | e <- testStrings]
-- Result: True

propMonadErrorCatchThrow :: Bool
propMonadErrorCatchThrow =
    let h = \e -> return (length e) :: Except String Int
    in allHold [runExcept (catchE (throwE e) h) == runExcept (h e) | e <- testStrings]
-- Result: True

propMonadErrorCatchReturn :: Bool
propMonadErrorCatchReturn =
    let h = \_ -> return 0 :: Except String Int
    in allHold [runExcept (catchE (return a) h) == runExcept (return a :: Except String Int) | a <- testInts]
-- Result: True

-- ================================================================
-- RWS Combined Laws
-- ================================================================

-- RWS should behave like stacked Reader, Writer, State
propRWSEquivalentToStack :: Bool
propRWSEquivalentToStack =
    let rwsComp = do
            r <- ask
            s <- get
            tell [r + s]
            put (s + 1)
            return (r * s)

        -- Run the RWS version
        (result1, state1, output1) = runRWS rwsComp 2 3

    in result1 == 6 && state1 == 4 && output1 == [5]
-- Result: True

propRWSLocalPreservesState :: Bool
propRWSLocalPreservesState =
    let comp = do
            s1 <- get
            r1 <- local (*2) ask
            s2 <- get
            return (s1 == s2, r1)
        (result, _, _) = runRWS comp 5 10
    in fst result == True && snd result == 10
-- Result: True

propRWSCensorPreservesState :: Bool
propRWSCensorPreservesState =
    let comp = do
            put 100
            censor (map toUpper) (tell "hello")
            get
        (result, finalState, output) = runRWS comp () 0
    in result == 100 && finalState == 100 && output == "HELLO"
-- Result: True

-- ================================================================
-- Continuation Laws
-- ================================================================

-- callCC early exit
propContCallCCEarlyExit :: Bool
propContCallCCEarlyExit =
    let comp = callCC $ \k -> do
            _ <- k 10
            return 20  -- Never reached
    in evalCont comp == 10
-- Result: True

-- callCC without using continuation
propContCallCCNoExit :: Bool
propContCallCCNoExit =
    let comp = callCC $ \_ -> return 20
    in evalCont comp == 20
-- Result: True

-- Nested callCC
propContNestedCallCC :: Bool
propContNestedCallCC =
    let comp = callCC $ \k1 -> do
            x <- callCC $ \k2 -> do
                _ <- k1 100  -- Exit outer
                k2 50        -- Never reached
            return (x + 1)   -- Never reached
    in evalCont comp == 100
-- Result: True

-- ================================================================
-- Alternative/MonadPlus Laws for MaybeT
-- ================================================================

-- empty <|> x = x
propMaybeTAlternativeIdentityLeft :: Bool
propMaybeTAlternativeIdentityLeft =
    allHold [runIdentity (runMaybeT (empty <|> return x)) ==
             runIdentity (runMaybeT (return x :: MaybeT Identity Int)) | x <- testInts]
-- Result: True

-- x <|> empty = x
propMaybeTAlternativeIdentityRight :: Bool
propMaybeTAlternativeIdentityRight =
    allHold [runIdentity (runMaybeT (return x <|> empty)) ==
             runIdentity (runMaybeT (return x :: MaybeT Identity Int)) | x <- testInts]
-- Result: True

-- (x <|> y) <|> z = x <|> (y <|> z)
propMaybeTAlternativeAssociativity :: Bool
propMaybeTAlternativeAssociativity =
    let x = return 1 :: MaybeT Identity Int
        y = return 2 :: MaybeT Identity Int
        z = return 3 :: MaybeT Identity Int
    in runIdentity (runMaybeT ((x <|> y) <|> z)) == runIdentity (runMaybeT (x <|> (y <|> z)))
-- Result: True

-- ================================================================
-- Main
-- ================================================================

main :: IO ()
main = do
    putStrLn "=== Functor Laws ==="
    print propIdentityFunctorIdentity
    print propIdentityFunctorComposition
    print propReaderTFunctorIdentity
    print propReaderTFunctorComposition
    print propWriterTFunctorIdentity
    print propWriterTFunctorComposition
    print propStateTFunctorIdentity
    print propStateTFunctorComposition
    print propExceptTFunctorIdentity
    print propExceptTFunctorComposition
    print propMaybeTFunctorIdentity
    print propMaybeTFunctorComposition
    print propContTFunctorIdentity
    print propContTFunctorComposition

    putStrLn "=== Applicative Laws ==="
    print propIdentityApplicativeIdentity
    print propIdentityApplicativeHomomorphism
    print propIdentityApplicativeInterchange
    print propIdentityApplicativeComposition
    print propReaderTApplicativeIdentity
    print propReaderTApplicativeHomomorphism
    print propStateTApplicativeIdentity
    print propStateTApplicativeHomomorphism
    print propExceptTApplicativeIdentity
    print propExceptTApplicativeHomomorphism

    putStrLn "=== Monad Laws ==="
    print propIdentityMonadLeftIdentity
    print propIdentityMonadRightIdentity
    print propIdentityMonadAssociativity
    print propReaderTMonadLeftIdentity
    print propReaderTMonadRightIdentity
    print propReaderTMonadAssociativity
    print propWriterTMonadLeftIdentity
    print propWriterTMonadRightIdentity
    print propWriterTMonadAssociativity
    print propStateTMonadLeftIdentity
    print propStateTMonadRightIdentity
    print propStateTMonadAssociativity
    print propExceptTMonadLeftIdentity
    print propExceptTMonadRightIdentity
    print propExceptTMonadAssociativity
    print propMaybeTMonadLeftIdentity
    print propMaybeTMonadRightIdentity
    print propMaybeTMonadAssociativity
    print propContTMonadLeftIdentity
    print propContTMonadRightIdentity
    print propContTMonadAssociativity

    putStrLn "=== MonadTrans Laws ==="
    print propReaderTLiftReturn
    print propReaderTLiftBind
    print propStateTLiftReturn
    print propStateTLiftBind
    print propWriterTLiftReturn
    print propWriterTLiftBind
    print propExceptTLiftReturn
    print propExceptTLiftBind
    print propMaybeTLiftReturn
    print propMaybeTLiftBind

    putStrLn "=== MTL Class Laws ==="
    print propMonadReaderAskLocal
    print propMonadWriterTellMempty
    print propMonadWriterTellAppend
    print propMonadStateGetPut
    print propMonadStatePutGet
    print propMonadStatePutPut
    print propMonadErrorThrowBind
    print propMonadErrorCatchThrow
    print propMonadErrorCatchReturn

    putStrLn "=== RWS Laws ==="
    print propRWSEquivalentToStack
    print propRWSLocalPreservesState
    print propRWSCensorPreservesState

    putStrLn "=== Continuation Laws ==="
    print propContCallCCEarlyExit
    print propContCallCCNoExit
    print propContNestedCallCC

    putStrLn "=== Alternative/MonadPlus Laws ==="
    print propMaybeTAlternativeIdentityLeft
    print propMaybeTAlternativeIdentityRight
    print propMaybeTAlternativeAssociativity

    putStrLn "All property tests completed!"
