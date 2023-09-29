//===- FunctionSupport.h - Utility types for function-like ops --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines support types for Operations that represent function-like
// constructs to use.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_FUNCTIONINTERFACES_H
#define MLIR_IR_FUNCTIONINTERFACES_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallString.h"

namespace mlir {
class FunctionOpInterface;

namespace function_interface_impl {
/// Insert the specified arguments and update the function type attribute.
void insertFunctionArguments(FunctionOpInterface op,
                             ArrayRef<unsigned> argIndices, TypeRange argTypes,
                             ArrayRef<DictionaryAttr> argAttrs,
                             ArrayRef<Location> argLocs,
                             unsigned originalNumArgs, Type newType);

/// Insert the specified results and update the function type attribute.
void insertFunctionResults(FunctionOpInterface op,
                           ArrayRef<unsigned> resultIndices,
                           TypeRange resultTypes,
                           ArrayRef<DictionaryAttr> resultAttrs,
                           unsigned originalNumResults, Type newType);

/// Erase the specified arguments and update the function type attribute.
void eraseFunctionArguments(FunctionOpInterface op, const BitVector &argIndices,
                            Type newType);

/// Erase the specified results and update the function type attribute.
void eraseFunctionResults(FunctionOpInterface op,
                          const BitVector &resultIndices, Type newType);

/// Set a FunctionOpInterface operation's type signature.
void setFunctionType(FunctionOpInterface op, Type newType);

/// This function defines the internal implementation of the `verifyTrait`
/// method on CallableOpInterface::Trait.
template <typename ConcreteOp>
LogicalResult verifyTrait(ConcreteOp op) {
  if (failed(op.verifyType()))
    return failure();

  // Check that the op has exactly one region for the body.
  if (op->getNumRegions() != 1)
    return op.emitOpError("expects one region");

  return op.verifyBody();
}
} // namespace function_interface_impl
} // namespace mlir

//===----------------------------------------------------------------------===//
// Tablegen Interface Declarations
//===----------------------------------------------------------------------===//

#include "mlir/Interfaces/FunctionInterfaces.h.inc"

#endif // MLIR_IR_FUNCTIONINTERFACES_H
