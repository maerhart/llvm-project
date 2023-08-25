//===- BufferDeallocationOpInterfaceImpl.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SCF/Transforms/BufferDeallocationOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Bufferization/IR/BufferDeallocationOpInterface.h"

using namespace mlir;
using namespace mlir::bufferization;

namespace {
/// 
struct InParallelOpInterface
    : public BufferDeallocationOpInterface::ExternalModel<InParallelOpInterface,
                                                    scf::InParallelOp> {
  FailureOr<Operation *> process(Operation *op, DeallocationState &state,
                      const DeallocationOptions &options) const {
    auto inParallelOp = cast<scf::InParallelOp>(op);
    if (!inParallelOp.getRegion().front().empty())
      return op->emitError("only supported with empty region");

    // Collect the values to deallocate and retain and use them to create the
    // dealloc operation.
    Block *block = op->getBlock();
    SmallVector<Value> memrefs, conditions, toRetain;
    if (failed(state.getMemrefsAndConditionsToDeallocate(builder, op.getLoc(), block,
                                                  memrefs, conditions)))
      return failure();

    state.getMemrefsToRetain(block, nullptr, {}, toRetain);
    if (memrefs.empty() && toRetain.empty())
      return op.getOperation();

    auto deallocOp = builder.create<bufferization::DeallocOp>(
        op.getLoc(), memrefs, conditions, toRetain);

    // We want to replace the current ownership of the retained values with the
    // result values of the dealloc operation as they are always unique.
    state.resetOwnerships(deallocOp.getRetained(), block);
    for (auto [retained, ownership] :
        llvm::zip(deallocOp.getRetained(), deallocOp.getUpdatedConditions()))
      state.updateOwnership(retained, ownership, block);

    return op;
  }
};

} // namespace

void mlir::scf::registerBufferDeallocationOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, SCFDialect *dialect) {
    InParallelOp::attachInterface<InParallelOpInterface>(*ctx);
  });
}
