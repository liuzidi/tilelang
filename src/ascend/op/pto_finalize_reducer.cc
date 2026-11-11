// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.
#include "pto_finalize_reducer.h"
#include "op/utils.h"
namespace tvm {
namespace tl {
PtoFinalizeReducerOp::PtoFinalizeReducerOp(
    ffi::Array<PrimExpr> args,
    ffi::Map<ffi::String, ffi::ObjectRef> annotations) {
  ICHECK(args.size() == 3 || args.size() == 4)
      << "pto_finalize_reducer expects accumulator, result, op and optional "
         "seed";
  auto n = ffi::make_object<PtoFinalizeReducerOpNode>();
  auto acc = NormalizeToAccessRegion(args[0], kAccessRead);
  auto dst = NormalizeToAccessRegion(args[1], kAccessWrite);
  ICHECK_EQ(acc.access_mask, kAccessRead);
  ICHECK_EQ(dst.access_mask, kAccessWrite);
  n->accumulator = acc.region;
  n->result = dst.region;
  const auto *kind = args[2].as<IntImmNode>();
  ICHECK(kind && kind->value >= 0 && kind->value <= 2)
      << "pto_finalize_reducer supports sum/max/min";
  n->op = static_cast<ReducerV2OpType>(kind->value);
  auto steps = annotations.Get("plan_steps");
  ICHECK(steps.has_value()) << "pto_finalize_reducer requires plan_steps";
  auto array = steps.value().as<ffi::Array<ffi::Any>>();
  ICHECK(array.has_value())
      << "pto_finalize_reducer plan_steps must be integers";
  for (const auto &entry : array.value()) {
    auto step = entry.try_cast<Integer>();
    ICHECK(step.has_value())
        << "pto_finalize_reducer plan_steps must be IntImm integers";
    n->plan_steps.push_back(step.value());
  }
  ICHECK_EQ(n->plan_steps.size() % 2, 0)
      << "plan_steps requires width/scale pairs";
  for (const auto &step : n->plan_steps)
    ICHECK_GT(step->value, 0) << "plan_steps must be positive";
  std::vector<AccessRegion> accesses{acc, dst};
  if (args.size() == 4) {
    auto seed = NormalizeToAccessRegion(args[3], kAccessRead);
    ICHECK_EQ(seed.access_mask, kAccessRead);
    n->seed = seed.region;
    accesses.push_back(seed);
  }
  n->SetAccessRegions(accesses);
  data_ = std::move(n);
}
Stmt PtoFinalizeReducerOpNode::Lower(const LowerArgs &,
                                     arith::Analyzer *) const {
  LOG(FATAL)
      << "pto_finalize_reducer must be consumed by VectorizeParallelToPTO";
}
LayoutMap PtoFinalizeReducerOpNode::InferLayout(const LayoutInferArgs &,
                                                InferLevel) const {
  return {};
}
TileOperator PtoFinalizeReducerOpNode::Clone() const {
  return TileOperator(ffi::make_object<PtoFinalizeReducerOpNode>(*this));
}
TIR_REGISTER_TL_TILE_OP(PtoFinalizeReducerOp, pto_finalize_reducer)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));
} // namespace tl
} // namespace tvm
