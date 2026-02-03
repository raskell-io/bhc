module Middle where

import Base

addTwo x = increment (increment x)
subTwo x = decrement (decrement x)
