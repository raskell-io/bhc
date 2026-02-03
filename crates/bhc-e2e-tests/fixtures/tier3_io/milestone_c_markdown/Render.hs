module Render where

import Types

-- | Render a full HTML document from blocks.
renderDocument blocks =
  "<html>\n<body>\n" ++ renderBlocks blocks ++ "</body>\n</html>\n"

-- | Render a list of blocks to HTML.
renderBlocks blocks = case blocks of
  [] -> ""
  (block : rest) ->
    case block of
      BlankLine -> renderBlocks rest
      _ -> renderBlock block ++ "\n" ++ renderBlocks rest

-- | Render a single block to HTML.
renderBlock block = case block of
  Heading level text ->
    let tag = headingTag level
    in "<" ++ tag ++ ">" ++ escapeHtml text ++ "</" ++ tag ++ ">"
  Paragraph text ->
    "<p>" ++ escapeHtml text ++ "</p>"
  CodeBlock code ->
    "<pre><code>" ++ escapeHtml code ++ "</code></pre>"
  UnorderedList items ->
    "<ul>\n" ++ renderListItems items ++ "</ul>"
  OrderedList items ->
    "<ol>\n" ++ renderListItems items ++ "</ol>"
  Blockquote text ->
    "<blockquote><p>" ++ escapeHtml text ++ "</p></blockquote>"
  HRule -> "<hr>"
  BlankLine -> ""

-- | Get HTML tag for heading level.
headingTag level = case level of
  1 -> "h1"
  2 -> "h2"
  3 -> "h3"
  4 -> "h4"
  5 -> "h5"
  _ -> "h6"

-- | Render list items.
renderListItems items = case items of
  [] -> ""
  (item : rest) ->
    "<li>" ++ escapeHtml item ++ "</li>\n" ++ renderListItems rest

-- | Escape a string for HTML.
escapeHtml s = case s of
  [] -> ""
  (c : rest) -> escapeChar c ++ escapeHtml rest

-- | Escape a single character for HTML.
escapeChar c = case c of
  '&' -> "&amp;"
  '<' -> "&lt;"
  '>' -> "&gt;"
  '"' -> "&quot;"
  _ -> [c]

