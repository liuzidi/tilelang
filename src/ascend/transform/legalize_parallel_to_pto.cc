// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.
#include "parallel_to_pto_utils.h"
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>
namespace tvm {
namespace tl {
using namespace tirx;
using namespace tirx::transform;
namespace {
class ReducerConditionalMutator : public StmtExprMutator {
public:
  pto::ParallelStorageQuery storage;
  pto::PtoReducerStates states;
  bool in_parallel{false};
  Stmt VisitStmt_(const SBlockNode *op) final {
    auto saved = states;
    if (op->name_hint == "SIMD_VF")
      states = pto::CollectReducerStates(ffi::GetRef<SBlock>(op), storage);
    Stmt result = StmtExprMutator::VisitStmt_(op);
    states = std::move(saved);
    return result;
  }
  Stmt VisitStmt_(const ForNode *op) final {
    bool saved = in_parallel;
    in_parallel |= op->kind == ForKind::kParallel;
    Stmt result = StmtExprMutator::VisitStmt_(op);
    in_parallel = saved;
    return result;
  }
  Stmt VisitStmt_(const IfThenElseNode *op) final {
    if (in_parallel && !states.accumulators.empty())
      return pto::LegalizeReducerConditional(op, states);
    return StmtExprMutator::VisitStmt_(op);
  }
};
} // namespace
tvm::transform::Pass LegalizeParallelToPTO() {
  auto pass_func = [](PrimFunc func, const IRModule &,
                      const tvm::transform::PassContext &) {
    auto target = func->GetAttr<Target>(tvm::attr::kTarget);
    if (!target.defined() || !TargetIsPTO(target.value()))
      return func;
    ReducerConditionalMutator mutator;
    mutator.storage.Build(func->body);
    func.CopyOnWrite()->body = mutator(func->body);
    return func;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.LegalizeParallelToPTO", {});
}
TVM_FFI_STATIC_INIT_BLOCK() {
  tvm::ffi::reflection::GlobalDef().def("tl.transform.LegalizeParallelToPTO",
                                        LegalizeParallelToPTO);
}
} // namespace tl
} // namespace tvm
