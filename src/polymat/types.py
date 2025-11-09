from typing import Annotated, Callable, Literal

import numpy as np
from numpy.typing import ArrayLike

# Mechanical quantity arrays
type Scalar = np.float64

type Vector = Annotated[ArrayLike, Literal["n"]]

type Tensor = Annotated[ArrayLike, Literal[3, 3]]


# Stress computation function signatures
type ElasticModel = Callable[[Tensor, list[float]], Tensor]
