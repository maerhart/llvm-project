//===- BufferAllocationAnalysis.h -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of a local stateless buffer allocation
// analysis. This analysis walks from the values being compared to determine
// whether they refer to the same originally allocated buffer.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_BUFFERIZATION_TRANSFORMS_BUFFERALLOCATIONANALYSIS_H_
#define MLIR_DIALECT_BUFFERIZATION_TRANSFORMS_BUFFERALLOCATIONANALYSIS_H_

namespace mlir {
class Value;
class Operation;

namespace bufferization {
/// The possible results of an alias query.
class AllocationResult {
public:
  enum Kind {
    /// The two locations do not alias at all.
    ///
    /// This value is arranged to convert to false, while all other values
    /// convert to true. This allows a boolean context to convert the result to
    /// a binary flag indicating whether there is the possibility of aliasing.
    Different = 0,
    /// The two locations may or may not alias. This is the least precise
    /// result.
    MaybeSame,
    /// The two locations precisely alias each other.
    Same,
  };

  AllocationResult(Kind kind) : kind(kind) {}
  bool operator==(const AllocationResult &other) const {
    return kind == other.kind;
  }
  bool operator!=(const AllocationResult &other) const {
    return !(*this == other);
  }

  /// Allow conversion to bool to signal if there is an aliasing or not.
  explicit operator bool() const { return kind != Different; }

  /// Merge this alias result with `other` and return a new result that
  /// represents the conservative merge of both results. If the results
  /// represent a known alias, the stronger alias is chosen (i.e.
  /// Partial+Must=Must). If the two results are conflicting, MayAlias is
  /// returned.
  AllocationResult merge(AllocationResult other) const;

  /// Returns if this result indicates no possibility of aliasing.
  bool isDifferent() const { return kind == Different; }

  /// Returns if this result is a may alias.
  bool isMaybeSame() const { return kind == MaybeSame; }

  /// Returns if this result is a must alias.
  bool isSame() const { return kind == Same; }

private:
  /// The internal kind of the result.
  Kind kind;
};

/// This class implements a local form of alias analysis that tries to identify
/// the underlying values addressed by each value and performs a few basic
/// checks to see if they alias.
class BufferAllocationAnalysis {
public:
  BufferAllocationAnalysis(Operation *op) {}
  virtual ~BufferAllocationAnalysis() = default;

  /// Given two values, return their aliasing behavior.
  AllocationResult alias(Value lhs, Value rhs);

protected:
  /// Given the two values, return their aliasing behavior.
  virtual AllocationResult aliasImpl(Value lhs, Value rhs);
};
} // namespace bufferization
} // namespace mlir

#endif // MLIR_DIALECT_BUFFERIZATION_TRANSFORMS_BUFFERALLOCATIONANALYSIS_H_
