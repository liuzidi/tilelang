// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.
#ifndef TVM_TL_ASCEND_OP_PTO_FINALIZE_REDUCER_H_
#define TVM_TL_ASCEND_OP_PTO_FINALIZE_REDUCER_H_
#include "op/reducer.h"
namespace tvm {
namespace tl {
class PtoFinalizeReducerOpNode : public TileOperatorNode {
public:
  BufferRegion accumulator;
  BufferRegion result;
  ffi::Optional<BufferRegion> seed;
  ReducerV2OpType op;
  ffi::Array<Integer> plan_steps;
  Stmt Lower(const LowerArgs &, arith::Analyzer *) const override;
  LayoutMap InferLayout(const LayoutInferArgs &, InferLevel) const override;
  TileOperator Clone() const override;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.PtoFinalizeReducerOp",
                                    PtoFinalizeReducerOpNode, TileOperatorNode);
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<PtoFinalizeReducerOpNode>()
        .def_ro("accumulator", &PtoFinalizeReducerOpNode::accumulator)
        .def_ro("result", &PtoFinalizeReducerOpNode::result)
        .def_ro("seed", &PtoFinalizeReducerOpNode::seed)
        .def_ro("plan_steps", &PtoFinalizeReducerOpNode::plan_steps);
  }
};
class PtoFinalizeReducerOp : public TileOperator {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(PtoFinalizeReducerOp, TileOperator,
                                             PtoFinalizeReducerOpNode);
  PtoFinalizeReducerOp(ffi::Array<PrimExpr> args,
                       ffi::Map<ffi::String, ffi::ObjectRef> annotations);
  static const Op &Get();
};
} // namespace tl
} // namespace tvm
#endif
