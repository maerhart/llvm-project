// DEFINE: %{canonicalize} = -canonicalize=enable-patterns="bufferization-skip-extract-metadata-of-alloc,bufferization-erase-always-false-dealloc,bufferization-erase-empty-dealloc,bufferization-dealloc-remove-duplicate-retained-memrefs,bufferization-dealloc-remove-duplicate-dealloc-memrefs",region-simplify=false

// RUN: mlir-opt -verify-diagnostics -buffer-deallocation \
// RUN:   %{canonicalize} -buffer-deallocation-simplification %{canonicalize} -split-input-file %s | FileCheck %s

func.func @parallel_insert_slice_no_conflict(%arg0: index, %arg1: index, %arg2: memref<?xf32, strided<[?], offset: ?>>, %arg3: memref<?xf32, strided<[?], offset: ?>>) -> f32 {
  %cst = arith.constant 4.200000e+01 : f32
  %c0 = arith.constant 0 : index
  scf.forall (%arg4) in (%arg1) {
    %subview = memref.subview %arg3[5] [%arg0] [1] : memref<?xf32, strided<[?], offset: ?>> to memref<?xf32, strided<[?], offset: ?>>
    linalg.fill ins(%cst : f32) outs(%subview : memref<?xf32, strided<[?], offset: ?>>)
  }
  %0 = memref.load %arg3[%c0] : memref<?xf32, strided<[?], offset: ?>>
  return %0 : f32
}

// CHECK-LABEL: func @parallel_insert_slice_no_conflict
//  CHECK-SAME: (%arg0: index, %arg1: index, %arg2: memref<?xf32, strided<[?], offset: ?>>, %arg3: memref<?xf32, strided<[?], offset: ?>>)
