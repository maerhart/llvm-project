//===- FunctionSupport.cpp - Utility types for function-like ops ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/CallInterfaces.h"

using namespace mlir;
using namespace mlir::callable_interface_impl;

//===----------------------------------------------------------------------===//
// Tablegen Interface Definitions
//===----------------------------------------------------------------------===//

#include "mlir/Interfaces/FunctionInterfaces.cpp.inc"

void function_interface_impl::insertFunctionArguments(
    FunctionOpInterface op, ArrayRef<unsigned> argIndices, TypeRange argTypes,
    ArrayRef<DictionaryAttr> argAttrs, ArrayRef<Location> argLocs,
    unsigned originalNumArgs, Type newType) {
  assert(argIndices.size() == argTypes.size());
  assert(argIndices.size() == argAttrs.size() || argAttrs.empty());
  assert(argIndices.size() == argLocs.size());
  if (argIndices.empty())
    return;

  // There are 3 things that need to be updated:
  // - Function type.
  // - Arg attrs.
  // - Block arguments of entry block.
  Block &entry = op->getRegion(0).front();

  // Update the argument attributes of the function.
  ArrayAttr oldArgAttrs = op.getArgAttrsAttr();
  if (oldArgAttrs || !argAttrs.empty()) {
    SmallVector<DictionaryAttr, 4> newArgAttrs;
    newArgAttrs.reserve(originalNumArgs + argIndices.size());
    unsigned oldIdx = 0;
    auto migrate = [&](unsigned untilIdx) {
      if (!oldArgAttrs) {
        newArgAttrs.resize(newArgAttrs.size() + untilIdx - oldIdx);
      } else {
        auto oldArgAttrRange = oldArgAttrs.getAsRange<DictionaryAttr>();
        newArgAttrs.append(oldArgAttrRange.begin() + oldIdx,
                           oldArgAttrRange.begin() + untilIdx);
      }
      oldIdx = untilIdx;
    };
    for (unsigned i = 0, e = argIndices.size(); i < e; ++i) {
      migrate(argIndices[i]);
      newArgAttrs.push_back(argAttrs.empty() ? DictionaryAttr{} : argAttrs[i]);
    }
    migrate(originalNumArgs);
    setAllArgAttrDicts(op, newArgAttrs);
  }

  // Update the function type and any entry block arguments.
  op.setFunctionTypeAttr(TypeAttr::get(newType));
  for (unsigned i = 0, e = argIndices.size(); i < e; ++i)
    entry.insertArgument(argIndices[i] + i, argTypes[i], argLocs[i]);
}

void function_interface_impl::insertFunctionResults(
    FunctionOpInterface op, ArrayRef<unsigned> resultIndices,
    TypeRange resultTypes, ArrayRef<DictionaryAttr> resultAttrs,
    unsigned originalNumResults, Type newType) {
  assert(resultIndices.size() == resultTypes.size());
  assert(resultIndices.size() == resultAttrs.size() || resultAttrs.empty());
  if (resultIndices.empty())
    return;

  // There are 2 things that need to be updated:
  // - Function type.
  // - Result attrs.

  // Update the result attributes of the function.
  ArrayAttr oldResultAttrs = op.getResAttrsAttr();
  if (oldResultAttrs || !resultAttrs.empty()) {
    SmallVector<DictionaryAttr, 4> newResultAttrs;
    newResultAttrs.reserve(originalNumResults + resultIndices.size());
    unsigned oldIdx = 0;
    auto migrate = [&](unsigned untilIdx) {
      if (!oldResultAttrs) {
        newResultAttrs.resize(newResultAttrs.size() + untilIdx - oldIdx);
      } else {
        auto oldResultAttrsRange = oldResultAttrs.getAsRange<DictionaryAttr>();
        newResultAttrs.append(oldResultAttrsRange.begin() + oldIdx,
                              oldResultAttrsRange.begin() + untilIdx);
      }
      oldIdx = untilIdx;
    };
    for (unsigned i = 0, e = resultIndices.size(); i < e; ++i) {
      migrate(resultIndices[i]);
      newResultAttrs.push_back(resultAttrs.empty() ? DictionaryAttr{}
                                                   : resultAttrs[i]);
    }
    migrate(originalNumResults);
    setAllResultAttrDicts(op, newResultAttrs);
  }

  // Update the function type.
  op.setFunctionTypeAttr(TypeAttr::get(newType));
}

void function_interface_impl::eraseFunctionArguments(
    FunctionOpInterface op, const BitVector &argIndices, Type newType) {
  // There are 3 things that need to be updated:
  // - Function type.
  // - Arg attrs.
  // - Block arguments of entry block.
  Block &entry = op->getRegion(0).front();

  // Update the argument attributes of the function.
  if (ArrayAttr argAttrs = op.getArgAttrsAttr()) {
    SmallVector<DictionaryAttr, 4> newArgAttrs;
    newArgAttrs.reserve(argAttrs.size());
    for (unsigned i = 0, e = argIndices.size(); i < e; ++i)
      if (!argIndices[i])
        newArgAttrs.emplace_back(llvm::cast<DictionaryAttr>(argAttrs[i]));
    setAllArgAttrDicts(op, newArgAttrs);
  }

  // Update the function type and any entry block arguments.
  op.setFunctionTypeAttr(TypeAttr::get(newType));
  entry.eraseArguments(argIndices);
}

void function_interface_impl::eraseFunctionResults(
    FunctionOpInterface op, const BitVector &resultIndices, Type newType) {
  // There are 2 things that need to be updated:
  // - Function type.
  // - Result attrs.

  // Update the result attributes of the function.
  if (ArrayAttr resAttrs = op.getResAttrsAttr()) {
    SmallVector<DictionaryAttr, 4> newResultAttrs;
    newResultAttrs.reserve(resAttrs.size());
    for (unsigned i = 0, e = resultIndices.size(); i < e; ++i)
      if (!resultIndices[i])
        newResultAttrs.emplace_back(llvm::cast<DictionaryAttr>(resAttrs[i]));
    setAllResultAttrDicts(op, newResultAttrs);
  }

  // Update the function type.
  op.setFunctionTypeAttr(TypeAttr::get(newType));
}

//===----------------------------------------------------------------------===//
// Function type signature.
//===----------------------------------------------------------------------===//

void function_interface_impl::setFunctionType(FunctionOpInterface op,
                                              Type newType) {
  unsigned oldNumArgs = op.getNumArguments();
  unsigned oldNumResults = op.getNumResults();
  op.setFunctionTypeAttr(TypeAttr::get(newType));
  unsigned newNumArgs = op.getNumArguments();
  unsigned newNumResults = op.getNumResults();

  // Functor used to update the argument and result attributes of the function.
  auto emptyDict = DictionaryAttr::get(op.getContext());
  auto updateAttrFn = [&](auto isArg, unsigned oldCount, unsigned newCount) {
    constexpr bool isArgVal = std::is_same_v<decltype(isArg), std::true_type>;

    if (oldCount == newCount)
      return;
    // The new type has no arguments/results, just drop the attribute.
    if (newCount == 0) {
      if (isArgVal)
        op.removeArgAttrsAttr();
      else
        op.removeResAttrsAttr();
    }

    ArrayAttr attrs;
    if (isArgVal)
      attrs = op.getArgAttrsAttr();
    else
      attrs = op.getResAttrsAttr();
    if (!attrs)
      return;

    // The new type has less arguments/results, take the first N attributes.
    if (newCount < oldCount) {
      if (isArgVal)
        return setAllArgAttrDicts(op, attrs.getValue().take_front(newCount));
      return setAllResultAttrDicts(op, attrs.getValue().take_front(newCount));
    }

    // Otherwise, the new type has more arguments/results. Initialize the new
    // arguments/results with empty dictionary attributes.
    SmallVector<Attribute> newAttrs(attrs.begin(), attrs.end());
    newAttrs.resize(newCount, emptyDict);
    if (isArgVal)
      setAllArgAttrDicts(op, newAttrs);
    else
      setAllResultAttrDicts(op, newAttrs);
  };

  // Update the argument and result attributes.
  updateAttrFn(std::true_type{}, oldNumArgs, newNumArgs);
  updateAttrFn(std::false_type{}, oldNumResults, newNumResults);
}
