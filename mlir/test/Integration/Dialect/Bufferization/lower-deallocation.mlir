// RUN: mlir-opt %s -pass-pipeline="builtin.module(func.func(canonicalize,convert-scf-to-cf),convert-vector-to-llvm,expand-strided-metadata,lower-affine,convert-arith-to-llvm,finalize-memref-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)" | \

// RUN: mlir-cpu-runner -O3 -e main -entry-point-result=void \
// RUN:   -shared-libs=%mlir_runner_utils,%mlir_c_runner_utils
// COM: RUN: FileCheck %s

func.func @helper(%m1: index, %m2: index, %o1: i1, %o2: i1, %r1: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %alloc_5 = memref.alloca() : memref<2xindex>
  %alloc_6 = memref.alloca() : memref<2xi1>
  %alloc_7 = memref.alloca() : memref<1xindex>
  memref.store %m1, %alloc_5[%c0] : memref<2xindex>
  memref.store %m2, %alloc_5[%c1] : memref<2xindex>
  memref.store %o1, %alloc_6[%c0] : memref<2xi1>
  memref.store %o2, %alloc_6[%c1] : memref<2xi1>
  memref.store %r1, %alloc_7[%c0] : memref<1xindex>
  %cast = memref.cast %alloc_5 : memref<2xindex> to memref<?xindex>
  %cast_10 = memref.cast %alloc_6 : memref<2xi1> to memref<?xi1>
  %cast_11 = memref.cast %alloc_7 : memref<1xindex> to memref<?xindex>
  %alloc_12 = memref.alloca() : memref<2xi1>
  %alloc_13 = memref.alloca() : memref<1xi1>
  %cast_14 = memref.cast %alloc_12 : memref<2xi1> to memref<?xi1>
  %cast_15 = memref.cast %alloc_13 : memref<1xi1> to memref<?xi1>
  call @dealloc_helper(%cast, %cast_11, %cast_10, %cast_14, %cast_15) : (memref<?xindex>, memref<?xindex>, memref<?xi1>, memref<?xi1>, memref<?xi1>) -> ()

  %should_dealloc = memref.cast %cast_14#0 : memref<?xi1> to memref<*xi1>
  %new_ownerships = memref.cast %cast_15#0 : memref<?xi1> to memref<*xi1>

  call @printMemrefI1(%should_dealloc) : (memref<*xi1>) -> ()
  call @printMemrefI1(%new_ownerships) : (memref<*xi1>) -> ()

  return
}

func.func @main() {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %true = arith.constant true
  %false = arith.constant false

  call @helper(%c1, %c2, %true, %true, %c1) : (index, index, i1, i1, index) -> ()

  return
}
func.func private @printMemrefI1(memref<*xi1>) attributes {llvm.emit_c_interface}
func.func @dealloc_helper(%arg0: memref<?xindex>, %arg1: memref<?xindex>, %arg2: memref<?xi1>, %arg3: memref<?xi1>, %arg4: memref<?xi1>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %true = arith.constant true
  %false = arith.constant false
  %dim = memref.dim %arg0, %c0 : memref<?xindex>
  %dim_0 = memref.dim %arg1, %c0 : memref<?xindex>
  scf.for %arg5 = %c0 to %dim_0 step %c1 {
    memref.store %false, %arg4[%arg5] : memref<?xi1>
  }
  scf.for %arg5 = %c0 to %dim step %c1 {
    %0 = memref.load %arg0[%arg5] : memref<?xindex>
    %1 = memref.load %arg2[%arg5] : memref<?xi1>
    %2 = scf.for %arg6 = %c0 to %dim_0 step %c1 iter_args(%arg7 = %true) -> (i1) {
      %5 = memref.load %arg1[%arg6] : memref<?xindex>
      %6 = arith.cmpi eq, %5, %0 : index
      scf.if %6 {
        %9 = memref.load %arg4[%arg6] : memref<?xi1>
        %10 = arith.ori %9, %1 : i1
        memref.store %10, %arg4[%arg6] : memref<?xi1>
      }
      %7 = arith.cmpi ne, %5, %0 : index
      %8 = arith.andi %arg7, %7 : i1
      scf.yield %8 : i1
    }
    %3 = scf.for %arg6 = %c0 to %arg5 step %c1 iter_args(%arg7 = %2) -> (i1) {
      %5 = memref.load %arg0[%arg6] : memref<?xindex>
      %6 = arith.cmpi ne, %5, %0 : index
      %7 = arith.andi %arg7, %6 : i1
      scf.yield %7 : i1
    }
    %4 = arith.andi %3, %1 : i1
    memref.store %4, %arg3[%arg5] : memref<?xi1>
  }
  return
}
