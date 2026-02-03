module Types where

-- Block-level elements
data Block
  = Heading Int String
  | Paragraph String
  | CodeBlock String
  | UnorderedList [String]
  | OrderedList [String]
  | Blockquote String
  | HRule
  | BlankLine

