//===- CallInterfaces.cpp - ControlFlow Interfaces ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Interfaces/CallInterfaces.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// CallOpInterface
//===----------------------------------------------------------------------===//

/// Resolve the callable operation for given callee to a CallableOpInterface, or
/// nullptr if a valid callable was not resolved. `symbolTable` is an optional
/// parameter that will allow for using a cached symbol table for symbol lookups
/// instead of performing an O(N) scan.
Operation *
CallOpInterface::resolveCallable(SymbolTableCollection *symbolTable) {
  CallInterfaceCallable callable = getCallableForCallee();
  if (auto symbolVal = dyn_cast<Value>(callable))
    return symbolVal.getDefiningOp();

  // If the callable isn't a value, lookup the symbol reference.
  auto symbolRef = callable.get<SymbolRefAttr>();
  if (symbolTable)
    return symbolTable->lookupNearestSymbolFrom(getOperation(), symbolRef);
  return SymbolTable::lookupNearestSymbolFrom(getOperation(), symbolRef);
}

//===----------------------------------------------------------------------===//
// CallInterfaces
//===----------------------------------------------------------------------===//

#include "mlir/Interfaces/CallInterfaces.cpp.inc"

//===----------------------------------------------------------------------===//
// Function Arguments and Results.
//===----------------------------------------------------------------------===//

static bool isEmptyAttrDict(Attribute attr) {
  return llvm::cast<DictionaryAttr>(attr).empty();
}

DictionaryAttr callable_interface_impl::getArgAttrDict(CallableOpInterface op,
                                                       unsigned index) {
  ArrayAttr attrs = op.getArgAttrsAttr();
  DictionaryAttr argAttrs =
      attrs ? llvm::cast<DictionaryAttr>(attrs[index]) : DictionaryAttr();
  return argAttrs;
}

DictionaryAttr
callable_interface_impl::getResultAttrDict(CallableOpInterface op,
                                           unsigned index) {
  ArrayAttr attrs = op.getResAttrsAttr();
  DictionaryAttr resAttrs =
      attrs ? llvm::cast<DictionaryAttr>(attrs[index]) : DictionaryAttr();
  return resAttrs;
}

ArrayRef<NamedAttribute>
callable_interface_impl::getArgAttrs(CallableOpInterface op, unsigned index) {
  auto argDict = getArgAttrDict(op, index);
  return argDict ? argDict.getValue() : std::nullopt;
}

ArrayRef<NamedAttribute>
callable_interface_impl::getResultAttrs(CallableOpInterface op,
                                        unsigned index) {
  auto resultDict = getResultAttrDict(op, index);
  return resultDict ? resultDict.getValue() : std::nullopt;
}

/// Get either the argument or result attributes array.
template <bool isArg>
static ArrayAttr getArgResAttrs(CallableOpInterface op) {
  if constexpr (isArg)
    return op.getArgAttrsAttr();
  else
    return op.getResAttrsAttr();
}

/// Set either the argument or result attributes array.
template <bool isArg>
static void setArgResAttrs(CallableOpInterface op, ArrayAttr attrs) {
  if constexpr (isArg)
    op.setArgAttrsAttr(attrs);
  else
    op.setResAttrsAttr(attrs);
}

/// Erase either the argument or result attributes array.
template <bool isArg>
static void removeArgResAttrs(CallableOpInterface op) {
  if constexpr (isArg)
    op.removeArgAttrsAttr();
  else
    op.removeResAttrsAttr();
}

/// Set all of the argument or result attribute dictionaries for a function.
template <bool isArg>
static void setAllArgResAttrDicts(CallableOpInterface op,
                                  ArrayRef<Attribute> attrs) {
  if (llvm::all_of(attrs, isEmptyAttrDict))
    removeArgResAttrs<isArg>(op);
  else
    setArgResAttrs<isArg>(op, ArrayAttr::get(op->getContext(), attrs));
}

void callable_interface_impl::setAllArgAttrDicts(
    CallableOpInterface op, ArrayRef<DictionaryAttr> attrs) {
  setAllArgAttrDicts(op, ArrayRef<Attribute>(attrs.data(), attrs.size()));
}

void callable_interface_impl::setAllArgAttrDicts(CallableOpInterface op,
                                                 ArrayRef<Attribute> attrs) {
  auto wrappedAttrs = llvm::map_range(attrs, [op](Attribute attr) -> Attribute {
    return !attr ? DictionaryAttr::get(op->getContext()) : attr;
  });
  setAllArgResAttrDicts</*isArg=*/true>(op, llvm::to_vector<8>(wrappedAttrs));
}

void callable_interface_impl::setAllResultAttrDicts(
    CallableOpInterface op, ArrayRef<DictionaryAttr> attrs) {
  setAllResultAttrDicts(op, ArrayRef<Attribute>(attrs.data(), attrs.size()));
}

void callable_interface_impl::setAllResultAttrDicts(CallableOpInterface op,
                                                    ArrayRef<Attribute> attrs) {
  auto wrappedAttrs = llvm::map_range(attrs, [op](Attribute attr) -> Attribute {
    return !attr ? DictionaryAttr::get(op->getContext()) : attr;
  });
  setAllArgResAttrDicts</*isArg=*/false>(op, llvm::to_vector<8>(wrappedAttrs));
}

/// Update the given index into an argument or result attribute dictionary.
template <bool isArg>
static void setArgResAttrDict(CallableOpInterface op, unsigned numTotalIndices,
                              unsigned index, DictionaryAttr attrs) {
  ArrayAttr allAttrs = getArgResAttrs<isArg>(op);
  if (!allAttrs) {
    if (attrs.empty())
      return;

    // If this attribute is not empty, we need to create a new attribute array.
    SmallVector<Attribute, 8> newAttrs(numTotalIndices,
                                       DictionaryAttr::get(op->getContext()));
    newAttrs[index] = attrs;
    setArgResAttrs<isArg>(op, ArrayAttr::get(op->getContext(), newAttrs));
    return;
  }
  // Check to see if the attribute is different from what we already have.
  if (allAttrs[index] == attrs)
    return;

  // If it is, check to see if the attribute array would now contain only empty
  // dictionaries.
  ArrayRef<Attribute> rawAttrArray = allAttrs.getValue();
  if (attrs.empty() &&
      llvm::all_of(rawAttrArray.take_front(index), isEmptyAttrDict) &&
      llvm::all_of(rawAttrArray.drop_front(index + 1), isEmptyAttrDict))
    return removeArgResAttrs<isArg>(op);

  // Otherwise, create a new attribute array with the updated dictionary.
  SmallVector<Attribute, 8> newAttrs(rawAttrArray.begin(), rawAttrArray.end());
  newAttrs[index] = attrs;
  setArgResAttrs<isArg>(op, ArrayAttr::get(op->getContext(), newAttrs));
}

void callable_interface_impl::setArgAttrs(CallableOpInterface op,
                                          unsigned index,
                                          ArrayRef<NamedAttribute> attributes) {
  assert(index < op.getNumArguments() && "invalid argument number");
  return setArgResAttrDict</*isArg=*/true>(
      op, op.getNumArguments(), index,
      DictionaryAttr::get(op->getContext(), attributes));
}

void callable_interface_impl::setArgAttrs(CallableOpInterface op,
                                          unsigned index,
                                          DictionaryAttr attributes) {
  return setArgResAttrDict</*isArg=*/true>(
      op, op.getNumArguments(), index,
      attributes ? attributes : DictionaryAttr::get(op->getContext()));
}

void callable_interface_impl::setResultAttrs(
    CallableOpInterface op, unsigned index,
    ArrayRef<NamedAttribute> attributes) {
  assert(index < op.getNumResults() && "invalid result number");
  return setArgResAttrDict</*isArg=*/false>(
      op, op.getNumResults(), index,
      DictionaryAttr::get(op->getContext(), attributes));
}

void callable_interface_impl::setResultAttrs(CallableOpInterface op,
                                             unsigned index,
                                             DictionaryAttr attributes) {
  assert(index < op.getNumResults() && "invalid result number");
  return setArgResAttrDict</*isArg=*/false>(
      op, op.getNumResults(), index,
      attributes ? attributes : DictionaryAttr::get(op->getContext()));
}
