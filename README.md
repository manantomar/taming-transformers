Original repo - https://github.com/CompVis/taming-transformers

## Loopy Vector Quantization

Implemented in - `taming/modules/vqvae` within the `LoopyVectorQuantizer()` class.

The idea is to create a mechanism for converting a vector into an arbitrary length of sequence of vectors. For this we use vector quantization as a potential solution, but turn into a loopy computation. 

Given a set of basis vectors (potentially learnable) in codebook like in standard VQ, an input vector can be mapped to a sequence of vectors with the following logic:

1. If the input vector is close enough to one of the basis vectors, given a certain threshold value, map it to that vector. Every new output in the loop will then follow a dummy vector once a close enough basis vector is found.

2. If the input vector is not within the threshold, then transform this vector, bringing it closer to whichever basis vector its most closest to, and place a dummy vector as the first vector in the output sequence. Go back and iterate again.

With this logic, a given vector, say `A = [0.5 1.2 0.3]`, will be mapped to a sequence of vectors, say `O = D, D, A'` where `D` denotes a dummy vector and `A'` denotes the closest vector to `A` in the codebook. The fact that two dummy vectors follow the output sequence `O` means that the original vector `A` has been transformed 2 times before it could be mapped within a certain threshold to one of the codebook vectors. 

This procedure gives us a simple way to create variable length outputs from a single vector and thus can be used in parallel for different encoding vectors at different sequence positions in the input to a transformer. 

## TODO

1. The above mechansim is simple to understand and encourages reuse of the basis vectors at the expense of variable length of the output sequence. However, it does so quite wastefully since every output sequence will have a certain number of dummy vectors, denoting how many transformations were made to bring the input vector close to one of the basis vectors. Such a sequence is highly compressible. Ex. `O = D, D, D, D, A'` can be compressed to `D 4 A'`, denoting `4` dummy vectors followed by `A'`. It is easy to see that the expressibility here is not the best. We could do better: say by following step 2), we transform the vector in such a manner that it can now be considered as a new vector that is compared with *all* the basis vectors again (instead of always being closest to `A'`). 
