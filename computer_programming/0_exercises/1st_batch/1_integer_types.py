"""
Create a variable storing the number 100 as a 16-bit integer. Then another variable which stores 100 as a 16-bit
unsigned integer (generally abbreviated “uint”). From the lecture notes, compute which are the minimum and
maximum values that these two types can store.

Verify your computations by “testing the boundaries”: perform
operations that reach the boundaries, then go beyond and observe that the numbers start cycling around.
Also experiment with mixing different number types and observe what happens (e.g. if you sum a signed integer
with an unsigned integer, etc.). Observe the output result, but also the output type. Put other integer types in the
mix too, e.g. 64-bit integers, signed and unsigned, standard python integers (try both small and very large values).
If you can, try to infer some (potentially vague) general underlying principle that is used by Python to decide which
output type to use if you mix different input types
"""

import numpy as np

a = np.int8(100)  # signed int 16bits => 2^15 - 1
b = np.uint16(10)  # unsigned 16 bits => 2^16 - 1

print(2**15 - 1)
print()
print(type(a))

# boh it auto converts to the higher memory type
