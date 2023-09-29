//===- CallInterfaces.h - Call Interfaces for MLIR --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the definitions of the call interfaces defined in
// `CallInterfaces.td`.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_CALLINTERFACES_H
#define MLIR_INTERFACES_CALLINTERFACES_H

#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/PointerUnion.h"

namespace mlir {
class CallableOpInterface;

/// A callable is either a symbol, or an SSA value, that is referenced by a
/// call-like operation. This represents the destination of the call.
struct CallInterfaceCallable : public PointerUnion<SymbolRefAttr, Value> {
  using PointerUnion<SymbolRefAttr, Value>::PointerUnion;
};

namespace callable_interface_impl {

/// Returns the dictionary attribute corresponding to the argument at 'index'.
/// If there are no argument attributes at 'index', a null attribute is
/// returned.
DictionaryAttr getArgAttrDict(CallableOpInterface op, unsigned index);

/// Returns the dictionary attribute corresponding to the result at 'index'.
/// If there are no result attributes at 'index', a null attribute is
/// returned.
DictionaryAttr getResultAttrDict(CallableOpInterface op, unsigned index);

/// Return all of the attributes for the argument at 'index'.
ArrayRef<NamedAttribute> getArgAttrs(CallableOpInterface op, unsigned index);

/// Return all of the attributes for the result at 'index'.
ArrayRef<NamedAttribute> getResultAttrs(CallableOpInterface op, unsigned index);

/// Set all of the argument or result attribute dictionaries for a function. The
/// size of `attrs` is expected to match the number of arguments/results of the
/// given `op`.
void setAllArgAttrDicts(CallableOpInterface op, ArrayRef<DictionaryAttr> attrs);
void setAllArgAttrDicts(CallableOpInterface op, ArrayRef<Attribute> attrs);
void setAllResultAttrDicts(CallableOpInterface op,
                           ArrayRef<DictionaryAttr> attrs);
void setAllResultAttrDicts(CallableOpInterface op, ArrayRef<Attribute> attrs);

//===----------------------------------------------------------------------===//
// Function Argument Attribute.
//===----------------------------------------------------------------------===//

/// Set the attributes held by the argument at 'index'.
void setArgAttrs(CallableOpInterface op, unsigned index,
                 ArrayRef<NamedAttribute> attributes);
void setArgAttrs(CallableOpInterface op, unsigned index,
                 DictionaryAttr attributes);

/// If the an attribute exists with the specified name, change it to the new
/// value. Otherwise, add a new attribute with the specified name/value.
template <typename ConcreteType>
void setArgAttr(ConcreteType op, unsigned index, StringAttr name,
                Attribute value) {
  NamedAttrList attributes(op.getArgAttrDict(index));
  Attribute oldValue = attributes.set(name, value);

  // If the attribute changed, then set the new arg attribute list.
  if (value != oldValue)
    op.setArgAttrs(index, attributes.getDictionary(value.getContext()));
}

/// Remove the attribute 'name' from the argument at 'index'. Returns the
/// removed attribute, or nullptr if `name` was not a valid attribute.
template <typename ConcreteType>
Attribute removeArgAttr(ConcreteType op, unsigned index, StringAttr name) {
  // Build an attribute list and remove the attribute at 'name'.
  NamedAttrList attributes(op.getArgAttrDict(index));
  Attribute removedAttr = attributes.erase(name);

  // If the attribute was removed, then update the argument dictionary.
  if (removedAttr)
    op.setArgAttrs(index, attributes.getDictionary(removedAttr.getContext()));
  return removedAttr;
}

//===----------------------------------------------------------------------===//
// Function Result Attribute.
//===----------------------------------------------------------------------===//

/// Set the attributes held by the result at 'index'.
void setResultAttrs(CallableOpInterface op, unsigned index,
                    ArrayRef<NamedAttribute> attributes);
void setResultAttrs(CallableOpInterface op, unsigned index,
                    DictionaryAttr attributes);

/// If the an attribute exists with the specified name, change it to the new
/// value. Otherwise, add a new attribute with the specified name/value.
template <typename ConcreteType>
void setResultAttr(ConcreteType op, unsigned index, StringAttr name,
                   Attribute value) {
  NamedAttrList attributes(op.getResultAttrDict(index));
  Attribute oldAttr = attributes.set(name, value);

  // If the attribute changed, then set the new arg attribute list.
  if (oldAttr != value)
    op.setResultAttrs(index, attributes.getDictionary(value.getContext()));
}

/// Remove the attribute 'name' from the result at 'index'.
template <typename ConcreteType>
Attribute removeResultAttr(ConcreteType op, unsigned index, StringAttr name) {
  // Build an attribute list and remove the attribute at 'name'.
  NamedAttrList attributes(op.getResultAttrDict(index));
  Attribute removedAttr = attributes.erase(name);

  // If the attribute was removed, then update the result dictionary.
  if (removedAttr)
    op.setResultAttrs(index,
                      attributes.getDictionary(removedAttr.getContext()));
  return removedAttr;
}

/// This function defines the internal implementation of the `verifyTrait`
/// method on CallableOpInterface::Trait.
template <typename ConcreteOp>
LogicalResult verifyTrait(ConcreteOp op) {
  if (ArrayAttr allArgAttrs = op.getAllArgAttrs()) {
    unsigned numArgs = op.getNumArguments();
    if (allArgAttrs.size() != numArgs) {
      return op.emitOpError()
             << "expects argument attribute array to have the same number of "
                "elements as the number of function arguments, got "
             << allArgAttrs.size() << ", but expected " << numArgs;
    }
    for (unsigned i = 0; i != numArgs; ++i) {
      DictionaryAttr argAttrs =
          llvm::dyn_cast_or_null<DictionaryAttr>(allArgAttrs[i]);
      if (!argAttrs) {
        return op.emitOpError() << "expects argument attribute dictionary "
                                   "to be a DictionaryAttr, but got `"
                                << allArgAttrs[i] << "`";
      }

      // Verify that all of the argument attributes are dialect attributes, i.e.
      // that they contain a dialect prefix in their name.  Call the dialect, if
      // registered, to verify the attributes themselves.
      for (auto attr : argAttrs) {
        if (!attr.getName().strref().contains('.'))
          return op.emitOpError("arguments may only have dialect attributes");
        if (Dialect *dialect = attr.getNameDialect()) {
          if (failed(dialect->verifyRegionArgAttribute(op, /*regionIndex=*/0,
                                                       /*argIndex=*/i, attr)))
            return failure();
        }
      }
    }
  }
  if (ArrayAttr allResultAttrs = op.getAllResultAttrs()) {
    unsigned numResults = op.getNumResults();
    if (allResultAttrs.size() != numResults) {
      return op.emitOpError()
             << "expects result attribute array to have the same number of "
                "elements as the number of function results, got "
             << allResultAttrs.size() << ", but expected " << numResults;
    }
    for (unsigned i = 0; i != numResults; ++i) {
      DictionaryAttr resultAttrs =
          llvm::dyn_cast_or_null<DictionaryAttr>(allResultAttrs[i]);
      if (!resultAttrs) {
        return op.emitOpError() << "expects result attribute dictionary "
                                   "to be a DictionaryAttr, but got `"
                                << allResultAttrs[i] << "`";
      }

      // Verify that all of the result attributes are dialect attributes, i.e.
      // that they contain a dialect prefix in their name.  Call the dialect, if
      // registered, to verify the attributes themselves.
      for (auto attr : resultAttrs) {
        if (!attr.getName().strref().contains('.'))
          return op.emitOpError("results may only have dialect attributes");
        if (Dialect *dialect = attr.getNameDialect()) {
          if (failed(dialect->verifyRegionResultAttribute(op, /*regionIndex=*/0,
                                                          /*resultIndex=*/i,
                                                          attr)))
            return failure();
        }
      }
    }
  }

  return success();
}

} // namespace callable_interface_impl
} // namespace mlir

/// Include the generated interface declarations.
#include "mlir/Interfaces/CallInterfaces.h.inc"

namespace llvm {

// Allow llvm::cast style functions.
template <typename To>
struct CastInfo<To, mlir::CallInterfaceCallable>
    : public CastInfo<To, mlir::CallInterfaceCallable::PointerUnion> {};

template <typename To>
struct CastInfo<To, const mlir::CallInterfaceCallable>
    : public CastInfo<To, const mlir::CallInterfaceCallable::PointerUnion> {};

} // namespace llvm

#endif // MLIR_INTERFACES_CALLINTERFACES_H
