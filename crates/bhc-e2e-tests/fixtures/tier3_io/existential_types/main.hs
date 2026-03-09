{-# LANGUAGE ExistentialQuantification #-}

-- Existential with builtin class constraint
data Showable = forall a. Show a => MkShowable a

showIt :: Showable -> String
showIt (MkShowable x) = show x

-- Existential with user-defined class (the zentinel pattern)
class Describable d where
  describe :: d -> String

data SomeDescribable = forall d. Describable d => MkDesc d

data Foo = Foo
data Bar = Bar

instance Describable Foo where
  describe _ = "I am Foo"

instance Describable Bar where
  describe _ = "I am Bar"

-- Pattern match extracting the existential dictionary
showDesc :: SomeDescribable -> String
showDesc (MkDesc x) = describe x

main :: IO ()
main = do
  putStrLn (showIt (MkShowable (42 :: Int)))
  putStrLn (showDesc (MkDesc Foo))
  putStrLn (showDesc (MkDesc Bar))
