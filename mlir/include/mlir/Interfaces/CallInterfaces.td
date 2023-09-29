//===- CallInterfaces.td - Call Interfaces for ops ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a set of interfaces that can be used to define information
// related to call-like and callable operations. Each of which are defined along
// with the respective interface below.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_CALLINTERFACES
#define MLIR_INTERFACES_CALLINTERFACES

include "mlir/IR/OpBase.td"

// `CallInterfaceCallable`: This is a type used to represent a single callable
// region. A callable is either a symbol, or an SSA value, that is referenced by
// a call-like operation. This represents the destination of the call.

/// Interface for call-like operations.
def CallOpInterface : OpInterface<"CallOpInterface"> {
  let description = [{
    A call-like operation is one that transfers control from one sub-routine to
    another. These operations may be traditional direct calls `call @foo`, or
    indirect calls to other operations `call_indirect %foo`. An operation that
    uses this interface, must *not* also provide the `CallableOpInterface`.
  }];
  let cppNamespace = "::mlir";

  let methods = [
    InterfaceMethod<[{
        Returns the callee of this call-like operation. A `callee` is either a
        reference to a symbol, via SymbolRefAttr, or a reference to a defined
        SSA value. If the reference is an SSA value, the SSA value corresponds
        to a region of a lambda-like operation.
      }],
      "::mlir::CallInterfaceCallable", "getCallableForCallee"
    >,
    InterfaceMethod<[{
        Sets the callee of this call-like operation. A `callee` is either a
        reference to a symbol, via SymbolRefAttr, or a reference to a defined
        SSA value. The type of the `callee` is expected to be the same as the
        return type of `getCallableForCallee`, e.g., `callee` should be
        SymbolRefAttr for `func.call`.
      }],
      "void", "setCalleeFromCallable", (ins "::mlir::CallInterfaceCallable":$callee)
    >,
    InterfaceMethod<[{
        Returns the operands within this call that are used as arguments to the
        callee.
      }],
      "::mlir::Operation::operand_range", "getArgOperands"
    >,
    InterfaceMethod<[{
        Returns the operands within this call that are used as arguments to the
        callee as a mutable range.
      }],
      "::mlir::MutableOperandRange", "getArgOperandsMutable">,
  ];

  let extraClassDeclaration = [{
    /// Resolve the callable operation for given callee to a
    /// CallableOpInterface, or nullptr if a valid callable was not resolved.
    /// `symbolTable` is an optional parameter that will allow for using a
    /// cached symbol table for symbol lookups instead of performing an O(N)
    /// scan.
    Operation *resolveCallable(SymbolTableCollection *symbolTable = nullptr);
  }];
}

/// Interface for callable operations.
def CallableOpInterface : OpInterface<"CallableOpInterface"> {
  let description = [{
    A callable operation is one who represents a potential sub-routine, and may
    be a target for a call-like operation (those providing the CallOpInterface
    above). These operations may be traditional functional operation
    `func @foo(...)`, as well as function producing operations
    `%foo = dialect.create_function(...)`. These operations may only contain a
    single region, or subroutine.
  }];
  let cppNamespace = "::mlir";

  let methods = [
    InterfaceMethod<[{
        Returns the region on the current operation that is callable. This may
        return null in the case of an external callable object, e.g. an external
        function.
      }],
      "::mlir::Region *", "getCallableRegion">,
    InterfaceMethod<[{
      Returns the callable's argument types based exclusively on the type (to
      allow for this method may be called on function declarations).
    }],
    "::llvm::ArrayRef<::mlir::Type>", "getArgumentTypes">,
    InterfaceMethod<[{
      Returns the callable's result types based exclusively on the type (to
      allow for this method may be called on function declarations).
    }],
    "::llvm::ArrayRef<::mlir::Type>", "getResultTypes">,

    InterfaceMethod<[{
        Get the array of argument attribute dictionaries. The method should
        return an array attribute containing only dictionary attributes equal in
        number to the number of region arguments. Alternatively, the method can
        return null to indicate that the region has no argument attributes.
      }],
      "::mlir::ArrayAttr", "getArgAttrsAttr", (ins),
      /*methodBody=*/[{}], /*defaultImplementation=*/[{ return nullptr; }]>,
    InterfaceMethod<[{
        Get the array of result attribute dictionaries. The method should return
        an array attribute containing only dictionary attributes equal in number
        to the number of region results. Alternatively, the method can return
        null to indicate that the region has no result attributes.
      }],
      "::mlir::ArrayAttr", "getResAttrsAttr", (ins),
      /*methodBody=*/[{}], /*defaultImplementation=*/[{ return nullptr; }]>,
    InterfaceMethod<[{
      Set the array of argument attribute dictionaries.
    }],
    "void", "setArgAttrsAttr", (ins "::mlir::ArrayAttr":$attrs),
      /*methodBody=*/[{}], /*defaultImplementation=*/[{ return; }]>,
    InterfaceMethod<[{
      Set the array of result attribute dictionaries.
    }],
    "void", "setResAttrsAttr", (ins "::mlir::ArrayAttr":$attrs),
      /*methodBody=*/[{}], /*defaultImplementation=*/[{ return; }]>,
    InterfaceMethod<[{
      Remove the array of argument attribute dictionaries. This is the same as
      setting all argument attributes to an empty dictionary. The method should
      return the removed attribute.
    }],
    "::mlir::Attribute", "removeArgAttrsAttr", (ins),
      /*methodBody=*/[{}], /*defaultImplementation=*/[{ return nullptr; }]>,
    InterfaceMethod<[{
      Remove the array of result attribute dictionaries. This is the same as
      setting all result attributes to an empty dictionary. The method should
      return the removed attribute.
    }],
    "::mlir::Attribute", "removeResAttrsAttr", (ins),
      /*methodBody=*/[{}], /*defaultImplementation=*/[{ return nullptr; }]>,
  ];

  let extraSharedClassDeclaration = [{
    /// Returns the number of function arguments.
    unsigned getNumArguments() { return $_op.getArgumentTypes().size(); }

    /// Returns the number of function results.
    unsigned getNumResults() { return $_op.getResultTypes().size(); }

    //===------------------------------------------------------------------===//
    // Argument Attributes
    //===------------------------------------------------------------------===//

    /// Return all of the attributes for the argument at 'index'.
    ArrayRef<NamedAttribute> getArgAttrs(unsigned index) {
      return callable_interface_impl::getArgAttrs($_op, index);
    }

    /// Return an ArrayAttr containing all argument attribute dictionaries of
    /// this function, or nullptr if no arguments have attributes.
    ArrayAttr getAllArgAttrs() { return $_op.getArgAttrsAttr(); }

    /// Return all argument attributes of this function.
    void getAllArgAttrs(SmallVectorImpl<DictionaryAttr> &result) {
      if (ArrayAttr argAttrs = getAllArgAttrs()) {
        auto argAttrRange = argAttrs.template getAsRange<DictionaryAttr>();
        result.append(argAttrRange.begin(), argAttrRange.end());
      } else {
        result.append($_op.getNumArguments(),
                      DictionaryAttr::get(this->getOperation()->getContext()));
      }
    }

    /// Return the specified attribute, if present, for the argument at 'index',
    /// null otherwise.
    Attribute getArgAttr(unsigned index, StringAttr name) {
      auto argDict = getArgAttrDict(index);
      return argDict ? argDict.get(name) : nullptr;
    }
    Attribute getArgAttr(unsigned index, StringRef name) {
      auto argDict = getArgAttrDict(index);
      return argDict ? argDict.get(name) : nullptr;
    }

    template <typename AttrClass>
    AttrClass getArgAttrOfType(unsigned index, StringAttr name) {
      return ::llvm::dyn_cast_or_null<AttrClass>(getArgAttr(index, name));
    }
    template <typename AttrClass>
    AttrClass getArgAttrOfType(unsigned index, StringRef name) {
      return ::llvm::dyn_cast_or_null<AttrClass>(getArgAttr(index, name));
    }

    /// Set the attributes held by the argument at 'index'.
    void setArgAttrs(unsigned index, ArrayRef<NamedAttribute> attributes) {
      callable_interface_impl::setArgAttrs($_op, index, attributes);
    }

    /// Set the attributes held by the argument at 'index'. `attributes` may be
    /// null, in which case any existing argument attributes are removed.
    void setArgAttrs(unsigned index, DictionaryAttr attributes) {
      callable_interface_impl::setArgAttrs($_op, index, attributes);
    }
    void setAllArgAttrs(ArrayRef<DictionaryAttr> attributes) {
      assert(attributes.size() == $_op.getNumArguments());
      callable_interface_impl::setAllArgAttrDicts($_op, attributes);
    }
    void setAllArgAttrs(ArrayRef<Attribute> attributes) {
      assert(attributes.size() == $_op.getNumArguments());
      callable_interface_impl::setAllArgAttrDicts($_op, attributes);
    }
    void setAllArgAttrs(ArrayAttr attributes) {
      assert(attributes.size() == $_op.getNumArguments());
      $_op.setArgAttrsAttr(attributes);
    }

    /// If the an attribute exists with the specified name, change it to the new
    /// value. Otherwise, add a new attribute with the specified name/value.
    void setArgAttr(unsigned index, StringAttr name, Attribute value) {
      callable_interface_impl::setArgAttr($_op, index, name, value);
    }
    void setArgAttr(unsigned index, StringRef name, Attribute value) {
      setArgAttr(index,
                 StringAttr::get(this->getOperation()->getContext(), name),
                 value);
    }

    /// Remove the attribute 'name' from the argument at 'index'. Return the
    /// attribute that was erased, or nullptr if there was no attribute with
    /// such name.
    Attribute removeArgAttr(unsigned index, StringAttr name) {
      return callable_interface_impl::removeArgAttr($_op, index, name);
    }
    Attribute removeArgAttr(unsigned index, StringRef name) {
      return removeArgAttr(
          index, StringAttr::get(this->getOperation()->getContext(), name));
    }

    //===------------------------------------------------------------------===//
    // Result Attributes
    //===------------------------------------------------------------------===//

    /// Return all of the attributes for the result at 'index'.
    ArrayRef<NamedAttribute> getResultAttrs(unsigned index) {
      return callable_interface_impl::getResultAttrs($_op, index);
    }

    /// Return an ArrayAttr containing all result attribute dictionaries of this
    /// function, or nullptr if no result have attributes.
    ArrayAttr getAllResultAttrs() { return $_op.getResAttrsAttr(); }

    /// Return all result attributes of this function.
    void getAllResultAttrs(SmallVectorImpl<DictionaryAttr> &result) {
      if (ArrayAttr argAttrs = getAllResultAttrs()) {
        auto argAttrRange = argAttrs.template getAsRange<DictionaryAttr>();
        result.append(argAttrRange.begin(), argAttrRange.end());
      } else {
        result.append($_op.getNumResults(),
                      DictionaryAttr::get(this->getOperation()->getContext()));
      }
    }

    /// Return the specified attribute, if present, for the result at 'index',
    /// null otherwise.
    Attribute getResultAttr(unsigned index, StringAttr name) {
      auto argDict = getResultAttrDict(index);
      return argDict ? argDict.get(name) : nullptr;
    }
    Attribute getResultAttr(unsigned index, StringRef name) {
      auto argDict = getResultAttrDict(index);
      return argDict ? argDict.get(name) : nullptr;
    }

    template <typename AttrClass>
    AttrClass getResultAttrOfType(unsigned index, StringAttr name) {
      return ::llvm::dyn_cast_or_null<AttrClass>(getResultAttr(index, name));
    }
    template <typename AttrClass>
    AttrClass getResultAttrOfType(unsigned index, StringRef name) {
      return ::llvm::dyn_cast_or_null<AttrClass>(getResultAttr(index, name));
    }

    /// Set the attributes held by the result at 'index'.
    void setResultAttrs(unsigned index, ArrayRef<NamedAttribute> attributes) {
      callable_interface_impl::setResultAttrs($_op, index, attributes);
    }

    /// Set the attributes held by the result at 'index'. `attributes` may be
    /// null, in which case any existing argument attributes are removed.
    void setResultAttrs(unsigned index, DictionaryAttr attributes) {
      callable_interface_impl::setResultAttrs($_op, index, attributes);
    }
    void setAllResultAttrs(ArrayRef<DictionaryAttr> attributes) {
      assert(attributes.size() == $_op.getNumResults());
      callable_interface_impl::setAllResultAttrDicts(
        $_op, attributes);
    }
    void setAllResultAttrs(ArrayRef<Attribute> attributes) {
      assert(attributes.size() == $_op.getNumResults());
      callable_interface_impl::setAllResultAttrDicts(
        $_op, attributes);
    }
    void setAllResultAttrs(ArrayAttr attributes) {
      assert(attributes.size() == $_op.getNumResults());
      $_op.setResAttrsAttr(attributes);
    }

    /// If the an attribute exists with the specified name, change it to the new
    /// value. Otherwise, add a new attribute with the specified name/value.
    void setResultAttr(unsigned index, StringAttr name, Attribute value) {
      callable_interface_impl::setResultAttr($_op, index, name, value);
    }
    void setResultAttr(unsigned index, StringRef name, Attribute value) {
      setResultAttr(index,
                    StringAttr::get(this->getOperation()->getContext(), name),
                    value);
    }

    /// Remove the attribute 'name' from the result at 'index'. Return the
    /// attribute that was erased, or nullptr if there was no attribute with
    /// such name.
    Attribute removeResultAttr(unsigned index, StringAttr name) {
      return callable_interface_impl::removeResultAttr($_op, index, name);
    }

    /// Returns the dictionary attribute corresponding to the argument at
    /// 'index'. If there are no argument attributes at 'index', a null
    /// attribute is returned.
    DictionaryAttr getArgAttrDict(unsigned index) {
      assert(index < $_op.getNumArguments() && "invalid argument number");
      return callable_interface_impl::getArgAttrDict($_op, index);
    }

    /// Returns the dictionary attribute corresponding to the result at 'index'.
    /// If there are no result attributes at 'index', a null attribute is
    /// returned.
    DictionaryAttr getResultAttrDict(unsigned index) {
      assert(index < $_op.getNumResults() && "invalid result number");
      return callable_interface_impl::getResultAttrDict($_op, index);
    }
  }];

  let verify = "return callable_interface_impl::verifyTrait(cast<ConcreteOp>($_op));";
}

#endif // MLIR_INTERFACES_CALLINTERFACES
