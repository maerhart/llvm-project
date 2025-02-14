// RUN: mlir-opt %s -split-input-file | mlir-opt | FileCheck %s

// CHECK-LABEL: func private @sparse_1d_tensor(
// CHECK-SAME: tensor<32xf64, #sparse_tensor.encoding<{ lvlTypes = [ "compressed" ] }>>)
func.func private @sparse_1d_tensor(tensor<32xf64, #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>>)

// -----

#CSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed),
  posWidth = 64,
  crdWidth = 64
}>

// CHECK-LABEL: func private @sparse_csr(
// CHECK-SAME: tensor<?x?xf32, #sparse_tensor.encoding<{ lvlTypes = [ "dense", "compressed" ], posWidth = 64, crdWidth = 64 }>>)
func.func private @sparse_csr(tensor<?x?xf32, #CSR>)

// -----

#CSR_explicit = #sparse_tensor.encoding<{
  map = {l0, l1} (d0 = l0, d1 = l1) -> (l0 = d0 : dense, l1 = d1 : compressed)
}>

// CHECK-LABEL: func private @CSR_explicit(
// CHECK-SAME: tensor<?x?xf64, #sparse_tensor.encoding<{ lvlTypes = [ "dense", "compressed" ] }>>
func.func private @CSR_explicit(%arg0: tensor<?x?xf64, #CSR_explicit>) {
  return
}

// -----

#CSC = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d1 : dense, d0 : compressed),
  posWidth = 0,
  crdWidth = 0
}>

// CHECK-LABEL: func private @sparse_csc(
// CHECK-SAME: tensor<?x?xf32, #sparse_tensor.encoding<{ lvlTypes = [ "dense", "compressed" ], dimToLvl = affine_map<(d0, d1) -> (d1, d0)> }>>)
func.func private @sparse_csc(tensor<?x?xf32, #CSC>)

// -----

#DCSC = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d1 : compressed, d0 : compressed),
  posWidth = 0,
  crdWidth = 64
}>

// CHECK-LABEL: func private @sparse_dcsc(
// CHECK-SAME: tensor<?x?xf32, #sparse_tensor.encoding<{ lvlTypes = [ "compressed", "compressed" ], dimToLvl = affine_map<(d0, d1) -> (d1, d0)>, crdWidth = 64 }>>)
func.func private @sparse_dcsc(tensor<?x?xf32, #DCSC>)

// -----

#COO = #sparse_tensor.encoding<{
  lvlTypes = [ "compressed-nu-no", "singleton-no" ]
}>

// CHECK-LABEL: func private @sparse_coo(
// CHECK-SAME: tensor<?x?xf32, #sparse_tensor.encoding<{ lvlTypes = [ "compressed-nu-no", "singleton-no" ] }>>)
func.func private @sparse_coo(tensor<?x?xf32, #COO>)

// -----

#BCOO = #sparse_tensor.encoding<{
  lvlTypes = [ "dense", "compressed-hi-nu", "singleton" ]
}>

// CHECK-LABEL: func private @sparse_bcoo(
// CHECK-SAME: tensor<?x?x?xf32, #sparse_tensor.encoding<{ lvlTypes = [ "dense", "compressed-hi-nu", "singleton" ] }>>)
func.func private @sparse_bcoo(tensor<?x?x?xf32, #BCOO>)

// -----

#SortedCOO = #sparse_tensor.encoding<{
  lvlTypes = [ "compressed-nu", "singleton" ]
}>

// CHECK-LABEL: func private @sparse_sorted_coo(
// CHECK-SAME: tensor<10x10xf64, #sparse_tensor.encoding<{ lvlTypes = [ "compressed-nu", "singleton" ] }>>)
func.func private @sparse_sorted_coo(tensor<10x10xf64, #SortedCOO>)

// -----

#BCSR = #sparse_tensor.encoding<{
   lvlTypes = [ "compressed", "compressed", "dense", "dense" ],
   dimToLvl  = affine_map<(i, j) -> (i floordiv 2, j floordiv 3, i mod 2, j mod 3)>
}>

// CHECK-LABEL: func private @sparse_bcsr(
// CHECK-SAME: tensor<10x60xf64, #sparse_tensor.encoding<{ lvlTypes = [ "compressed", "compressed", "dense", "dense" ], dimToLvl = affine_map<(d0, d1) -> (d0 floordiv 2, d1 floordiv 3, d0 mod 2, d1 mod 3)> }>>
func.func private @sparse_bcsr(tensor<10x60xf64, #BCSR>)


// -----

#ELL = #sparse_tensor.encoding<{
  lvlTypes = [ "dense", "dense", "compressed" ],
  dimToLvl = affine_map<(i,j)[c] -> (c*4*i, i, j)>
}>

// CHECK-LABEL: func private @sparse_ell(
// CHECK-SAME: tensor<?x?xf64, #sparse_tensor.encoding<{ lvlTypes = [ "dense", "dense", "compressed" ], dimToLvl = affine_map<(d0, d1)[s0] -> (d0 * (s0 * 4), d0, d1)> }>>
func.func private @sparse_ell(tensor<?x?xf64, #ELL>)

// -----

#CSR_SLICE = #sparse_tensor.encoding<{
  lvlTypes = [ "dense", "compressed" ],
  dimSlices = [ (1, 4, 1), (1, 4, 2) ]
}>

// CHECK-LABEL: func private @sparse_slice(
// CHECK-SAME: tensor<?x?xf64, #sparse_tensor.encoding<{ lvlTypes = [ "dense", "compressed" ], dimSlices = [ (1, 4, 1), (1, 4, 2) ] }>>
func.func private @sparse_slice(tensor<?x?xf64, #CSR_SLICE>)

// -----

#CSR_SLICE = #sparse_tensor.encoding<{
  lvlTypes = [ "dense", "compressed" ],
  dimSlices = [ (1, 4, 1), (1, 4, 2) ]
}>

// CHECK-LABEL: func private @sparse_slice(
// CHECK-SAME: tensor<?x?xf64, #sparse_tensor.encoding<{ lvlTypes = [ "dense", "compressed" ], dimSlices = [ (1, 4, 1), (1, 4, 2) ] }>>
func.func private @sparse_slice(tensor<?x?xf64, #CSR_SLICE>)

// -----

#CSR_SLICE = #sparse_tensor.encoding<{
  lvlTypes = [ "dense", "compressed" ],
  dimSlices = [ (1, ?, 1), (?, 4, 2) ]
}>

// CHECK-LABEL: func private @sparse_slice(
// CHECK-SAME: tensor<?x?xf64, #sparse_tensor.encoding<{ lvlTypes = [ "dense", "compressed" ], dimSlices = [ (1, ?, 1), (?, 4, 2) ] }>>
func.func private @sparse_slice(tensor<?x?xf64, #CSR_SLICE>)

// -----

// TODO: It is probably better to use [dense, dense, 2:4] (see NV_24 defined using new syntax
// below) to encode a 2D matrix, but it would require dim2lvl mapping which is not ready yet.
// So we take the simple path for now.
#NV_24= #sparse_tensor.encoding<{
  lvlTypes = [ "dense", "compressed24" ],
}>

// CHECK-LABEL: func private @sparse_2_out_of_4(
// CHECK-SAME: tensor<?x?xf64, #sparse_tensor.encoding<{ lvlTypes = [ "dense", "compressed24" ] }>>
func.func private @sparse_2_out_of_4(tensor<?x?xf64, #NV_24>)

// -----

#BCSR = #sparse_tensor.encoding<{
  map = ( i, j ) ->
  ( i floordiv 2 : compressed,
    j floordiv 3 : compressed,
    i mod 2      : dense,
    j mod 3      : dense
  )
}>

// CHECK-LABEL: func private @BCSR(
// CHECK-SAME: tensor<?x?xf64, #sparse_tensor.encoding<{ lvlTypes = [ "compressed", "compressed", "dense", "dense" ], dimToLvl = affine_map<(d0, d1) -> (d0 floordiv 2, d1 floordiv 3, d0 mod 2, d1 mod 3)> }>>
func.func private @BCSR(%arg0: tensor<?x?xf64, #BCSR>) {
  return
}

// -----

#BCSR_explicit = #sparse_tensor.encoding<{
  map =
  {il, jl, ii, jj}
  ( i = il * 2 + ii,
    j = jl * 3 + jj
  ) ->
  ( il = i floordiv 2 : compressed,
    jl = j floordiv 3 : compressed,
    ii = i mod 2      : dense,
    jj = j mod 3      : dense
  )
}>

// CHECK-LABEL: func private @BCSR_explicit(
// CHECK-SAME: tensor<?x?xf64, #sparse_tensor.encoding<{ lvlTypes = [ "compressed", "compressed", "dense", "dense" ], dimToLvl = affine_map<(d0, d1) -> (d0 floordiv 2, d1 floordiv 3, d0 mod 2, d1 mod 3)> }>>
func.func private @BCSR_explicit(%arg0: tensor<?x?xf64, #BCSR_explicit>) {
  return
}

// -----

#NV_24 = #sparse_tensor.encoding<{
  map = ( i, j ) ->
  ( i            : dense,
    j floordiv 4 : dense,
    j mod 4      : compressed24
  )
}>

// CHECK-LABEL: func private @NV_24(
// CHECK-SAME: tensor<?x?xf64, #sparse_tensor.encoding<{ lvlTypes = [ "dense", "dense", "compressed24" ], dimToLvl = affine_map<(d0, d1) -> (d0, d1 floordiv 4, d1 mod 4)> }>>
func.func private @NV_24(%arg0: tensor<?x?xf64, #NV_24>) {
  return
}
