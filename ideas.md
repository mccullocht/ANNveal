# Ideas for improvements

## Graphs

* Full DiskANN with binary vectors for navigation and int7 vectors inline.
* ACORN flat - ACORN-1 variant with a flat graph
* ACORN-gamma - the full HNSW with many additional edges.
* Gorder graph reordering.

## Quantization

* int8 vectors? not sure if this is a real problem.
* alpha scoring adjustment for binary vectors. would this just be scalar int1?
* cheap navigational scoring using masked hamming scoring
  * mask high bit of a quantized value and compute hamming distance.
  * use adjusted hamming score when navigating during search, int4 or int8 for result set.
* t-digest for quantile tasks
  * faster and more complete scalar quantization
  * evenly divide regions in sbq