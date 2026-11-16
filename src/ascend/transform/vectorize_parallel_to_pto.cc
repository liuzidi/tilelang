// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file vectorize_parallel_to_pto.cc
 * \brief VectorizeParallelToPTO: rewrite verified T.Parallel loops inside
 * PTO SIMD_VF blocks into logical VMI code.
 *
 * Consumes the pto_parallel_loop_layout fragment written by
 * LayoutInference and verified by VerifyParallelToPTO. Per parallel unit:
 *   1. invert the lane fragment through the shared helper (the inverse
 *      layout's own Forward result is the single source of truth);
 *   2. build the chunk loop for q in [0, E/L), mapping the loop variable
 *      to q * L + lane;
 *   3. classify every access via the shared MemoryAccess analysis
 *      (continuous -> vload/vstore via tl.access_ptr(start, L, rw);
 *      lane-uniform -> scalar BufferLoad with lane bound to 0);
 *   4. convert value expressions (value roles tracked by result dtype);
 *      every VMI with a return value is bound through EmitBind before use;
 *   5. delete the consumed layout annotation, keep the SIMD_VF block
 *      annotations, and leave no unconverted Parallel behind.
 *
 * Supported shapes: 1D parallel units and two-layer adjacent Parallel nests
 * (2D — the nest converts as one unit with the layout
 * annotation on its outermost For); sequential units allowed; static
 * divisible extents; full masks. A third parallel layer or any other
 * unsupported structure is rejected by Verify.
 *
 * Non-divisible extents emit a tail chunk whose vload reads the full L
 * lanes from its chunk base: the ISA intentionally keeps load instructions
 * unmasked for performance, so the per-chunk tail mask constrains
 * computation and stores only, never the load. Reading the full L lanes
 * past the logical tail is still safe on device — the top 8K of UB is
 * reserved as stack space, so the extra read-only lanes cannot fault; they
 * carry undefined values that the compute and store masks keep out of the
 * valid outputs. Compact outputs of exactly E elements remain safe because
 * the masked store never writes past E.
 */

#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <map>
#include <set>
#include <sstream>
#include <unordered_map>
#include <vector>

#include "parallel_to_pto_utils.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;
namespace pto_analysis = ::tvm::tl::pto;

namespace {

DataType VectorDType(DataType elem, int64_t lanes) {
  return elem.with_lanes(static_cast<int>(lanes));
}

const Op &VmiOp(const std::string &tail) { return Op::Get("tl.vmi." + tail); }

// This version keeps local DCE in Vectorize so
// its directly emitted TIR/PTODSL is cleaner and easier to inspect.
// Cleanup may later move to PTOAS canonicalization/DCE to reduce
// TileLang-specific code and maintenance; clean-output checks would then
// target the optimized PTOAS IR. The emitted IR must be valid and preserve
// program semantics even before DCE.
//
// Dead vector-chain elimination on one converted chunk body. The
// conversion can leave Bind statements whose Var is never referenced:
// source binds consumed only through the address alias environment
// (bind_env_) never use their converted Var, and an unused source bind
// strands its whole value chain (vci/vbrc/vadd/vload).
//
// The traversal is a scope-aware *reverse liveness* pass — a
// dead Bind must not survive just because it was wrapped in an AttrStmt
// or appears as the only statement of a body. The rules are:
//   1. A SeqStmt is processed back to front; `live` always holds the
//      vars everything *after* the current statement needs.
//   2. A Bind's LHS var is a definition, never a use. When the result is
//      not live and the RHS is discardable, the Bind is deleted and its
//      RHS inputs do NOT become live (so the dead chain collapses in one
//      pass); otherwise it is kept and its RHS inputs join `live`.
//   3. An AttrStmt body is cleaned in its own nested scope with a fresh
//      liveness set; only the vars actually referenced inside propagate
//      outward. Attribute node/value references are always uses, and the
//      attribute structure and evaluation position are preserved.
//   4. A bare Bind follows the same rules; when everything dies the
//      remains are a legal no-op (Evaluate(0)).
// Nothing is merged, reordered or otherwise rescheduled;
// referenced loads stay exactly where they were (vecadd keeps its three
// vloads), and Bind-free (frontend-inlined) bodies are untouched.
class DeadBindElim {
public:
  Stmt operator()(const Stmt &stmt) {
    std::set<const VarNode *> live;
    Stmt out = Run(stmt, &live);
    return out.defined() ? out : NoOp();
  }

private:
  static Stmt NoOp() { return Evaluate(make_zero(DataType::Int(32))); }

  static void CollectExprVars(const PrimExpr &expr,
                              std::set<const VarNode *> *live) {
    PostOrderVisit(expr, [&](const ObjectRef &n) {
      if (const auto *v = n.as<VarNode>()) {
        live->insert(v);
      }
    });
  }

  static void CollectStmtVars(const Stmt &stmt,
                              std::set<const VarNode *> *live) {
    PostOrderVisit(stmt, [&](const ObjectRef &n) {
      if (const auto *v = n.as<VarNode>()) {
        live->insert(v);
      }
    });
  }

  static bool DiscardableRhs(const PrimExpr &rhs) {
    // Positive "allowed to discard" set: a Call may disappear
    // only when it is one of the tl.vmi.* ops this pass actually emits as
    // a Bind RHS and that is pure/read-only by construction — vload
    // (read), vbrc, vci, vadd, vmul and vcvt (pure), enumerated from the
    // ConvertUnit/VmiOp call sites — or the pure tl.access_ptr address
    // descriptor nested inside a vload operand (already whitelisted as a
    // pure index intrinsic by IsSupportedAddressExpr). vstore never
    // appears as a Bind RHS; any other Call — call_extern, other tl.* or
    // an unknown op — keeps its Bind. Plain UB BufferLoads are
    // discardable: they are ordinary reads with no write effect (the
    // scalar "sload" binds). That "ordinary" premise is enforced by
    // VerifyParallelToPTO, not assumed here: accesses to
    // tirx.volatile-annotated allocations and volatile_scope attributes
    // are rejected there, and opaque calls are already excluded
    // from addresses.
    static const std::set<std::string> kDiscardable = {
        "tl.vmi.vload",
        "tl.vmi.vbrc",
        "tl.vmi.vci",
        "tl.vmi.vadd",
        "tl.vmi.vsub",
        "tl.vmi.vmul",
        "tl.vmi.vdiv",
        "tl.vmi.vabs",
        "tl.vmi.vexp",
        "tl.vmi.vln",
        "tl.vmi.vsqrt",
        "tl.vmi.vcvt",
        "tl.vmi.vinterpret_cast",
        "tl.access_ptr",
    };
    bool discardable = true;
    PostOrderVisit(rhs, [&](const ObjectRef &n) {
      if (const auto *call = n.as<CallNode>()) {
        const auto *op = call->op.as<OpNode>();
        const std::string name = op != nullptr ? op->name : "";
        if (kDiscardable.count(name) == 0) {
          discardable = false;
        }
      }
    });
    return discardable;
  }

  // Returns the kept statement, or an *undefined* Stmt when the statement
  // was deleted (only a Bind can be deleted). `live` is threaded from
  // later statements back to earlier ones.
  Stmt Run(const Stmt &stmt, std::set<const VarNode *> *live) {
    if (const auto *bind = stmt.as<BindNode>()) {
      if (live->count(bind->var.get()) == 0 && DiscardableRhs(bind->value)) {
        // Dead and safe to drop: the RHS inputs stay out of `live`, so a
        // chain feeding only this Bind collapses in the same pass.
        return Stmt();
      }
      // Kept: classic liveness — subtract the definition, add the uses.
      live->erase(bind->var.get());
      CollectExprVars(bind->value, live);
      return stmt;
    }
    if (const auto *seq = stmt.as<SeqStmtNode>()) {
      Array<Stmt> kept;
      for (auto it = seq->seq.rbegin(); it != seq->seq.rend(); ++it) {
        Stmt sub = Run(*it, live);
        if (sub.defined()) {
          kept.push_back(sub);
        }
      }
      if (kept.empty()) {
        return NoOp();
      }
      // std::reverse does not accept the ffi Array iterators on this
      // toolchain; re-append in reverse instead.
      Array<Stmt> forward;
      for (auto it = kept.rbegin(); it != kept.rend(); ++it) {
        forward.push_back(*it);
      }
      return SeqStmt::Flatten(forward);
    }
    if (const auto *attr = stmt.as<AttrStmtNode>()) {
      // Nested scope: clean the body with a fresh liveness set, then
      // propagate only the references that survive to its entry (external
      // uses; vars defined and used only inside never escape because the
      // converted Vars are unique objects). Attribute node/value
      // references are always uses and keep the structure.
      std::set<const VarNode *> body_live;
      Stmt body = Run(attr->body, &body_live);
      if (!body.defined()) {
        body = NoOp();
      }
      live->insert(body_live.begin(), body_live.end());
      if (auto ne = attr->node.try_cast<ObjectRef>()) {
        if (const auto *pe = ne.value().as<PrimExprNode>()) {
          CollectExprVars(GetRef<PrimExpr>(pe), live);
        }
      }
      CollectExprVars(attr->value, live);
      if (body.same_as(attr->body)) {
        return stmt;
      }
      return AttrStmt(attr->node, attr->attr_key, attr->value, body);
    }
    // Any other statement is always kept; all its Var occurrences are
    // uses.
    CollectStmtVars(stmt, live);
    return stmt;
  }
};

class VectorizeParallelToPTOImpl : public StmtExprMutator {
public:
  static PrimFunc Substitute(PrimFunc f) {
    VectorizeParallelToPTOImpl mutator;
    mutator.storage_.Build(f->body);
    mutator.storage_.ValidateOwnerUses(f->body);
    PrimFuncNode *fptr = f.CopyOnWrite();
    fptr->body = mutator.VisitStmt(f->body);
    class RemoveOld : public StmtMutator {
    public:
      explicit RemoveOld(const pto::ParallelStorageQuery &q) : q(q) {}
      Stmt VisitStmt_(const SBlockNode *op) final {
        SBlock b = Downcast<SBlock>(StmtMutator::VisitStmt_(op));
        Array<Buffer> kept;
        for (const Buffer &buf : b->alloc_buffers)
          if (!q.IsPrivate(buf->data))
            kept.push_back(buf);
        b.CopyOnWrite()->alloc_buffers = kept;
        return b;
      }
      Stmt VisitStmt_(const AllocBufferNode *op) final {
        if (q.IsPrivate(op->buffer->data))
          return Evaluate(0);
        return GetRef<Stmt>(op);
      }
      const pto::ParallelStorageQuery &q;
    };
    fptr->body = RemoveOld(mutator.storage_)(fptr->body);
    return GetRef<PrimFunc>(fptr);
  }

private:
  // ------------------------------------------------------------------
  // Unified unit discovery (shared rules with Verify).
  // A SIMD_VF region contains zero or more *sequential* parallel units.
  // Each unit is the outermost parallel For of a 1D loop or of a
  // two-layer adjacent Parallel nest (2D); a third parallel layer (or a
  // Parallel under an unsupported wrapper) is rejected by Verify and
  // defensively rejected here too.
  // ------------------------------------------------------------------
  // ------------------------------------------------------------------
  // SIMD_VF block entry
  // ------------------------------------------------------------------
  Stmt VisitStmt_(const SBlockNode *op) final {
    if (op->name_hint != "SIMD_VF") {
      return StmtExprMutator::VisitStmt_(op);
    }
    // Shared region scan: the same implementation Verify uses, so
    // skip/reject/convert decisions cannot drift apart.
    const Stmt &region = op->body;
    pto_analysis::PtoRegionScan scan = pto_analysis::ScanPtoRegion(region);
    if (!scan.has_any_parallel && !scan.has_direct_reduce) {
      // Pure hand-written VMI region (possibly with control flow): skipped
      // by both passes; no lanes annotation needed.
      return StmtExprMutator::VisitStmt_(op);
    }
    ICHECK(!scan.has_unsupported_wrapper)
        << "[VectorizeParallelToPTO] internal error: unsupported wrapper "
           "structure passed Verify: "
        << scan.issue;

    auto saved_reducers = reducers_;
    auto saved_buffers = vf_buffers_;
    auto saved_bind = bind_env_;
    auto saved_values = value_map_;
    auto saved_infos = value_infos_;
    auto saved_masks = full_masks_;
    auto saved_prelude = mask_prelude_;
    int64_t saved_lanes = lanes_;
    reducers_ = pto::CollectReducerStates(GetRef<SBlock>(op), storage_);
    lanes_ = reducers_.lanes;
    vf_buffers_.clear();
    full_masks_.clear();
    mask_prelude_.clear();
    for (const auto &[data, info] : reducers_.storage_values) {
      if (info.kind != pto::ValueKind::kVmi)
        continue;
      Buffer old = storage_.Allocation(data);
      vf_buffers_.emplace(
          old->data,
          decl_buffer(old->shape, old->dtype.with_lanes(info.logical_lanes),
                      old->name + "_v",
                      info.logical_lanes == 1 ? "local.var" : "local"));
    }
    if (scan.has_any_parallel)
      shared_mask_ = GetFullMask(lanes_);
    Stmt body = ConvertRegion(region);
    auto wrap = [&](Stmt inner) {
      inner = SeqStmt::Flatten(mask_prelude_, inner);
      for (const auto &[old, replacement] : vf_buffers_) {
        inner = AttrStmt(
            replacement->data, attr::kVmiAllocKind,
            StringImm(replacement->dtype.lanes() == 1 ? "data" : "data_array"),
            inner);
      }
      return inner;
    };
    if (const auto *scope = body.as<AttrStmtNode>();
        scope && scope->attr_key == "tl.simdvf_scope")
      body = AttrStmt(scope->node, scope->attr_key, scope->value,
                      wrap(scope->body));
    else
      body = wrap(body);
    Array<Buffer> allocations;
    for (const auto &old : op->alloc_buffers) {
      auto it = vf_buffers_.find(old->data);
      allocations.push_back(it == vf_buffers_.end() ? old : it->second);
    }
    auto annotations = op->annotations;
    annotations.erase(attr::kPtoReducerStates);
    Stmt result =
        SBlock(op->iter_vars, op->reads, op->writes, op->name_hint, body,
               op->init, allocations, op->match_buffers, annotations);
    reducers_ = std::move(saved_reducers);
    vf_buffers_ = std::move(saved_buffers);
    bind_env_ = std::move(saved_bind);
    value_map_ = std::move(saved_values);
    value_infos_ = std::move(saved_infos);
    full_masks_ = std::move(saved_masks);
    mask_prelude_ = std::move(saved_prelude);
    lanes_ = saved_lanes;
    return result;
  }

  Map<Var, int> buffer_versions_;

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    auto saved = buffer_versions_;
    if (op->attr_key == "tl.buffer_version")
      for (const auto &[data, count] : Downcast<Map<Var, int>>(op->node))
        buffer_versions_.Set(data, count);
    Stmt result = StmtExprMutator::VisitStmt_(op);
    buffer_versions_ = std::move(saved);
    return result;
  }

  Var GetFullMask(int64_t lanes) {
    auto it = full_masks_.find(lanes);
    if (it != full_masks_.end())
      return it->second;
    Var mask("full_mask_" + std::to_string(lanes), DataType::Bool(lanes));
    mask_prelude_.push_back(
        Bind(mask, Call(mask.dtype(), VmiOp("create_mask"), {Integer(lanes)},
                        {{"size", Integer(lanes)}})));
    full_masks_.emplace(lanes, mask);
    return mask;
  }

  pto::ValueInfo Info(const PrimExpr &value) const {
    if (const auto *var = value.as<VarNode>()) {
      auto it = value_infos_.find(GetRef<Var>(var));
      if (it != value_infos_.end())
        return it->second;
    }
    return {value.dtype().lanes() > 1 ? pto::ValueKind::kVmi
                                      : pto::ValueKind::kScalar,
            value.dtype().lanes()};
  }

  PrimExpr ToVmiWidth(const PrimExpr &source, int64_t width, Array<Stmt> *pre) {
    auto info = Info(source);
    if (info.kind == pto::ValueKind::kVmi && info.logical_lanes == width)
      return source;
    ICHECK(info.logical_lanes == 1) << "unsupported PTO value width conversion";
    pto::CheckBroadcastDType(source.dtype().element_of(),
                             "PTO value broadcast");
    Map<String, ObjectRef> attrs{{"size", Integer(width)}};
    if (info.kind == pto::ValueKind::kVmi)
      attrs.Set("group", Integer(1));
    return EmitBind(
        "broadcast",
        Call(source.dtype().with_lanes(width), VmiOp("vbrc"), {source}, attrs),
        pre);
  }

  PrimExpr ReadRegionValue(const PrimExpr &source, Array<Stmt> *pre) {
    if (const auto *load = source.as<BufferLoadNode>()) {
      auto it = vf_buffers_.find(load->buffer->data);
      if (it != vf_buffers_.end()) {
        Var value = EmitBind("result_snapshot",
                             BufferLoad(it->second, load->indices), pre);
        value_infos_[value] = {pto::ValueKind::kVmi, it->second->dtype.lanes()};
        return value;
      }
      return EmitBind("scalar_snapshot", source, pre);
    }
    if (const auto *var = source.as<VarNode>()) {
      if (auto mapped = value_map_.Get(GetRef<Var>(var)))
        return mapped.value();
    }
    return source;
  }

  Stmt LowerDirectReduce(const Call &call, int64_t lanes) {
    pto_analysis::PtoDirectReducePlan plan =
        pto_analysis::PlanPtoDirectReduce(call, &analyzer_, buffer_versions_);
    auto i32 = [](int64_t value) { return IntImm(DataType::Int(32), value); };
    DataType vector_dtype = plan.accumulator_dtype.with_lanes(lanes);
    DataType mask_dtype = DataType::Bool(lanes);
    Buffer accumulator =
        tirx::decl_buffer({i32(1)}, vector_dtype, "reduce_acc", "local.var");
    Var chunk("reduce_chunk", DataType::Int(32));
    Var output("reduce_output", DataType::Int(32));
    Var reduction("reduce_index", DataType::Int(32));
    Var full_mask("reduce_domain", mask_dtype);
    Var active_mask("reduce_active", mask_dtype);
    auto mask = [&](const PrimExpr &active) {
      return Call(mask_dtype, VmiOp("create_mask"), {active},
                  {{"size", i32(lanes)}});
    };
    auto pointer = [&](const BufferRegion &region,
                       const Array<PrimExpr> &indices, int64_t extent, int rw) {
      return Call(DataType::Handle(), tl::access_ptr(),
                  {BufferLoad(region->buffer, indices), i32(extent), i32(rw)});
    };
    auto source_indices = [&](const PrimExpr &out, const PrimExpr &red) {
      Array<PrimExpr> indices;
      for (const Range &range : plan.source->region) {
        indices.push_back(range->min);
      }
      indices.Set(plan.source_prefix + plan.reduction_axis, red);
      if (plan.logical_rank == 2) {
        indices.Set(plan.source_prefix + plan.output_axis, out);
      }
      return indices;
    };
    auto destination_indices = [&](const PrimExpr &out) {
      Array<PrimExpr> indices;
      for (const Range &range : plan.destination->region) {
        indices.push_back(range->min);
      }
      indices.Set(plan.destination_output_axis, out);
      return indices;
    };
    auto load = [&](const PrimExpr &out, const PrimExpr &red,
                    Array<Stmt> *statements) {
      DataType src_dtype = plan.source->buffer->dtype;
      int64_t footprint =
          ((lanes + 2048 / src_dtype.bits() - 1) / (2048 / src_dtype.bits())) *
          (2048 / src_dtype.bits());
      PrimExpr value = EmitBind(
          "reduce_load",
          Call(src_dtype.with_lanes(lanes), VmiOp("vload"),
               {pointer(plan.source, source_indices(out, red), footprint, 1),
                i32(0)},
               {{"size", i32(lanes)}}),
          statements);
      if (src_dtype != plan.accumulator_dtype) {
        value =
            EmitBind("reduce_cast",
                     Call(vector_dtype, VmiOp("vcvt"), {value},
                          {{"to_dtype",
                            StringImm(TvmDtypeName(plan.accumulator_dtype))}}),
                     statements);
      }
      return value;
    };
    auto read_accumulator = [&]() { return BufferLoad(accumulator, {i32(0)}); };
    const char *fold_name = plan.operation == "sum"   ? "vadd"
                            : plan.operation == "max" ? "vmax"
                                                      : "vmin";
    Array<Stmt> body;
    if (plan.reduce_across_lanes) {
      PrimExpr first = load(output, i32(0), &body);
      body.push_back(BufferStore(accumulator, first, {i32(0)}));
      auto accumulate = [&](const PrimExpr &red, const PrimExpr &active,
                            bool keep_inactive) {
        Array<Stmt> update;
        PrimExpr next = load(output, red, &update);
        PrimExpr folded = EmitBind("reduce_fold",
                                   Call(vector_dtype, VmiOp(fold_name),
                                        {read_accumulator(), next, active}),
                                   &update);
        if (keep_inactive) {
          folded = EmitBind("reduce_merge",
                            Call(vector_dtype, VmiOp("vsel"),
                                 {active, folded, read_accumulator()}),
                            &update);
        }
        update.push_back(BufferStore(accumulator, folded, {i32(0)}));
        return SeqStmt::Flatten(update);
      };
      int64_t full_chunks = plan.reduction_extent / lanes;
      int64_t tail_lanes = plan.reduction_extent % lanes;
      // The seed already covers the first chunk. Complete chunks need no
      // remaining-lane calculation or merge with the previous accumulator.
      if (full_chunks > 1) {
        body.push_back(For(chunk, i32(1), i32(full_chunks - 1),
                           ForKind::kSerial,
                           accumulate(chunk * i32(lanes), full_mask, false)));
      }
      if (full_chunks > 0 && tail_lanes != 0) {
        body.push_back(Bind(active_mask, mask(i32(tail_lanes))));
        body.push_back(accumulate(i32(full_chunks * lanes), active_mask, true));
      }
      ffi::Map<ffi::String, ObjectRef> annotations;
      if (plan.operation == "sum" && plan.accumulator_dtype.is_float()) {
        annotations.Set("reassoc", IntImm(DataType::Bool(), 1));
      }
      const char *reduce_name = plan.operation == "sum"   ? "vcadd"
                                : plan.operation == "max" ? "vcmax"
                                                          : "vcmin";
      PrimExpr result =
          EmitBind("reduce_result",
                   Call(plan.accumulator_dtype, VmiOp(reduce_name),
                        {read_accumulator(), full_mask}, annotations),
                   &body);
      if (!plan.clear) {
        PrimExpr previous = EmitBind(
            "reduce_previous",
            Call(plan.accumulator_dtype, VmiOp("vload"),
                 {pointer(plan.destination, destination_indices(output), 1, 1),
                  i32(0)},
                 {{"size", i32(1)}}),
            &body);
        PrimExpr one = EmitBind("reduce_one",
                                Call(DataType::Bool(), VmiOp("create_mask"),
                                     {i32(1)}, {{"size", i32(1)}}),
                                &body);
        // Merge the previous output only after the cross-lane reduction;
        // seeding every lane with it would count it VL times for sum.
        result = EmitBind("reduce_update",
                          Call(plan.accumulator_dtype, VmiOp(fold_name),
                               {previous, result, one}),
                          &body);
      }
      body.push_back(Evaluate(Call(
          DataType::Void(), VmiOp("vstore"),
          {result, pointer(plan.destination, destination_indices(output), 1, 2),
           i32(0)})));
      Stmt output_loop = For(output, i32(0), i32(plan.output_extent),
                             ForKind::kSerial, SeqStmt::Flatten(body));
      Stmt compute = SeqStmt::Flatten(
          Bind(full_mask, mask(i32(std::min(plan.reduction_extent, lanes)))),
          output_loop);
      return SeqStmt::Flatten(AllocBuffer(accumulator),
                              AttrStmt(accumulator->data, "tl.vmi.alloc_kind",
                                       StringImm("data"), compute));
    }
    body.push_back(
        Bind(active_mask, mask(i32(plan.output_extent) - chunk * i32(lanes))));
    PrimExpr first = load(chunk * i32(lanes), i32(0), &body);
    body.push_back(BufferStore(accumulator, first, {i32(0)}));
    Array<Stmt> update;
    PrimExpr next = load(chunk * i32(lanes), reduction, &update);
    PrimExpr folded = EmitBind("reduce_fold",
                               Call(vector_dtype, VmiOp(fold_name),
                                    {read_accumulator(), next, active_mask}),
                               &update);
    update.push_back(BufferStore(accumulator, folded, {i32(0)}));
    if (plan.reduction_extent > 1) {
      body.push_back(For(reduction, i32(1), i32(plan.reduction_extent - 1),
                         ForKind::kSerial, SeqStmt::Flatten(update)));
    }
    if (!plan.clear) {
      int64_t physical_lanes = 2048 / plan.accumulator_dtype.bits();
      int64_t footprint =
          ((lanes + physical_lanes - 1) / physical_lanes) * physical_lanes;
      PrimExpr previous = EmitBind(
          "reduce_previous",
          Call(vector_dtype, VmiOp("vload"),
               {pointer(plan.destination,
                        destination_indices(chunk * i32(lanes)), footprint, 1),
                i32(0)},
               {{"size", i32(lanes)}}),
          &body);
      PrimExpr combined =
          EmitBind("reduce_update",
                   Call(vector_dtype, VmiOp(fold_name),
                        {previous, read_accumulator(), active_mask}),
                   &body);
      body.push_back(BufferStore(accumulator, combined, {i32(0)}));
    }
    body.push_back(Evaluate(
        Call(DataType::Void(), VmiOp("vstore"),
             {read_accumulator(),
              pointer(plan.destination, destination_indices(chunk * i32(lanes)),
                      lanes, 2),
              i32(0), active_mask})));
    // Full chunks and the tail use the same vector reduction. Loads may read
    // complete physical lanes; active_mask limits the fold and output store.
    int64_t chunks = (plan.output_extent + lanes - 1) / lanes;
    Stmt compute = For(chunk, i32(0), i32(chunks), ForKind::kSerial,
                       SeqStmt::Flatten(body));
    return SeqStmt::Flatten(AllocBuffer(accumulator),
                            AttrStmt(accumulator->data, "tl.vmi.alloc_kind",
                                     StringImm("data"), compute));
  }

  Stmt ConvertPtoFinalize(const PtoFinalizeReducerOp &op) {
    pto::AnalyzePtoFinalize(op, reducers_);
    Array<Stmt> out;
    Buffer acc = vf_buffers_.at(op->accumulator->buffer->data);
    auto value =
        EmitBind("partial_snapshot", BufferLoad(acc, {Integer(0)}), &out);
    const char *names[] = {"vcadd", "vcmax", "vcmin"};
    Map<String, ObjectRef> attrs{{"group", Integer(1)}};
    if (op->op == ReducerV2OpType::kSum)
      attrs.Set("reassoc", Bool(true));
    PrimExpr result = EmitBind("reduced_one",
                               Call(acc->dtype.element_of(),
                                    VmiOp(names[static_cast<int>(op->op)]),
                                    {value, GetFullMask(lanes_)}, attrs),
                               &out);
    if (op->seed.defined()) {
      auto seed = ReadRegionValue(
          BufferLoad(op->seed.value()->buffer, {Integer(0)}), &out);
      seed = ToVmiWidth(seed, 1, &out);
      const char *combine[] = {"vadd", "vmax", "vmin"};
      result = EmitBind("seed_combine",
                        Call(result.dtype(),
                             VmiOp(combine[static_cast<int>(op->op)]),
                             {result, seed, GetFullMask(1)}),
                        &out);
    }
    out.push_back(BufferStore(vf_buffers_.at(op->result->buffer->data), result,
                              {Integer(0)}));
    return SeqStmt::Flatten(out);
  }

  Stmt ConvertRegion(const Stmt &stmt) {
    if (const auto *seq = stmt.as<SeqStmtNode>()) {
      Array<Stmt> out;
      for (const auto &child : seq->seq)
        out.push_back(ConvertRegion(child));
      return SeqStmt::Flatten(out);
    }
    if (const auto *loop = stmt.as<ForNode>()) {
      auto saved_bind = bind_env_;
      auto saved_values = value_map_;
      auto saved_infos = value_infos_;
      Stmt result;
      if (loop->kind == ForKind::kParallel) {
        For unit = GetRef<For>(loop);
        auto mapping = pto::InvertPtoLaneLayout(unit);
        auto saved_unit = current_unit_;
        auto saved_promoted = promoted_;
        auto saved_mapping = unit_mapping_;
        auto saved_loop = loop_var_, saved_chunk = chunk_var_;
        auto saved_analysis = chunk_analysis_, saved_lane = lane_var_;
        auto saved_index = lane_indexed_expr_;
        auto saved_mask = mask_var_;
        auto saved_remaining = remaining_buffer_;
        auto saved_nchunks = nchunks_, saved_lanes = lanes_;
        result = ConvertUnit(unit, mapping);
        Array<Stmt> declarations;
        for (const auto &old : storage_.PrivateAllocations(unit))
          declarations.push_back(AllocBuffer(
              promoted_.at(old->data),
              {{attr::kVmiAllocKind,
                String(old.scope() == "local.var" ? "data" : "data_array")}}));
        declarations.push_back(result);
        result = SeqStmt::Flatten(declarations);
        current_unit_ = saved_unit;
        promoted_ = std::move(saved_promoted);
        unit_mapping_ = saved_mapping;
        loop_var_ = saved_loop;
        chunk_var_ = saved_chunk;
        chunk_analysis_ = saved_analysis;
        lane_var_ = saved_lane;
        lane_indexed_expr_ = saved_index;
        mask_var_ = saved_mask;
        remaining_buffer_ = saved_remaining;
        nchunks_ = saved_nchunks;
        lanes_ = saved_lanes;
      } else {
        For replacement = GetRef<For>(loop);
        replacement.CopyOnWrite()->body = ConvertRegion(loop->body);
        result = replacement;
      }
      bind_env_ = std::move(saved_bind);
      value_map_ = std::move(saved_values);
      value_infos_ = std::move(saved_infos);
      return result;
    }
    if (const auto *attr = stmt.as<AttrStmtNode>()) {
      auto saved_bind = bind_env_;
      auto saved_values = value_map_;
      auto saved_infos = value_infos_;
      auto saved_versions = buffer_versions_;
      if (attr->attr_key == "tl.buffer_version")
        for (const auto &[data, count] : Downcast<Map<Var, int>>(attr->node))
          buffer_versions_.Set(data, count);
      Stmt body = ConvertRegion(attr->body);
      buffer_versions_ = std::move(saved_versions);
      bind_env_ = std::move(saved_bind);
      value_map_ = std::move(saved_values);
      value_infos_ = std::move(saved_infos);
      return AttrStmt(attr->node, attr->attr_key, attr->value, body);
    }
    if (const auto *bind = stmt.as<BindNode>()) {
      Array<Stmt> out;
      auto value = ReadRegionValue(bind->value, &out);
      Var replacement(bind->var->name_hint, value.dtype());
      bind_env_.Set(bind->var, bind->value);
      value_map_.Set(bind->var, replacement);
      value_infos_[replacement] = Info(value);
      out.push_back(Bind(replacement, value));
      return SeqStmt::Flatten(out);
    }
    if (const auto *eval = stmt.as<EvaluateNode>()) {
      if (const auto *call = eval->value.as<CallNode>();
          call && pto::IsPtoDirectReduce(GetRef<Call>(call)))
        return LowerDirectReduce(GetRef<Call>(call), lanes_);
      if (const auto *call = eval->value.as<CallNode>();
          call && call->op.same_as(PtoFinalizeReducerOp::Get()))
        return ConvertPtoFinalize(
            Downcast<PtoFinalizeReducerOp>(ParseOperator(GetRef<Call>(call))));
      return stmt;
    }
    if (const auto *store = stmt.as<BufferStoreNode>()) {
      Array<Stmt> out;
      PrimExpr value = ReadRegionValue(store->value, &out);
      auto data = store->buffer->data;
      bool seed = false;
      for (const auto &site : reducers_.finalizes)
        seed |= site.op->seed.defined() &&
                site.op->seed.value()->buffer->data.same_as(data);
      if (seed && Info(value).kind == pto::ValueKind::kVmi &&
          !vf_buffers_.count(data))
        vf_buffers_.emplace(
            data, decl_buffer(store->buffer->shape, store->buffer->dtype,
                              store->buffer->name + "_v", "local.var"));
      auto target = vf_buffers_.find(data);
      if (target != vf_buffers_.end()) {
        value = ToVmiWidth(value, target->second->dtype.lanes(), &out);
        out.push_back(BufferStore(target->second, value, store->indices));
      } else if (Info(value).kind == pto::ValueKind::kVmi) {
        PrimExpr ptr = Call(DataType::Handle(), tl::access_ptr(),
                            {BufferLoad(store->buffer, store->indices),
                             Integer(1), Integer(2)});
        out.push_back(Evaluate(
            Call(DataType::Void(), VmiOp("vstore"), {value, ptr, Integer(0)})));
      } else
        out.push_back(BufferStore(store->buffer, value, store->indices,
                                  store->predicate));
      return SeqStmt::Flatten(out);
    }
    if (const auto *branch = stmt.as<IfThenElseNode>()) {
      auto saved_bind = bind_env_;
      auto saved_values = value_map_;
      auto saved_infos = value_infos_;
      Stmt yes = ConvertRegion(branch->then_case);
      bind_env_ = saved_bind;
      value_map_ = saved_values;
      value_infos_ = saved_infos;
      Optional<Stmt> no;
      if (branch->else_case.defined())
        no = ConvertRegion(branch->else_case.value());
      bind_env_ = std::move(saved_bind);
      value_map_ = std::move(saved_values);
      value_infos_ = std::move(saved_infos);
      return IfThenElse(branch->condition, yes, no);
    }
    if (const auto *block = stmt.as<SBlockNode>()) {
      auto saved_bind = bind_env_;
      auto saved_values = value_map_;
      auto saved_infos = value_infos_;
      SBlock copy = GetRef<SBlock>(block);
      copy.CopyOnWrite()->body = ConvertRegion(block->body);
      bind_env_ = std::move(saved_bind);
      value_map_ = std::move(saved_values);
      value_infos_ = std::move(saved_infos);
      return copy;
    }
    if (const auto *realize = stmt.as<SBlockRealizeNode>())
      return SBlockRealize(realize->iter_values, realize->predicate,
                           Downcast<SBlock>(ConvertRegion(realize->block)));
    return stmt;
  }

  // ------------------------------------------------------------------
  // Per-unit conversion
  // ------------------------------------------------------------------
  Stmt ConvertUnit(const For &loop,
                   const pto_analysis::PtoInverseMapping &mapping) {
    // The inverse mapping comes from the region entry: it is the shared
    // source of truth for lanes, the mask plan and the body conversion
    // (see InvertPtoLaneLayout).
    // Reset per-unit conversion state.
    current_unit_ = loop;
    promoted_.clear();
    for (const Buffer &old : storage_.PrivateAllocations(loop)) {
      storage_.CheckStorageV1(old);
      Buffer b = decl_buffer(old->shape, old->dtype.with_lanes(mapping.lanes),
                             old->name + "_v", old.scope());
      auto *w = b.CopyOnWrite();
      w->strides = old->strides;
      w->elem_offset = old->elem_offset;
      promoted_.emplace(old->data, b);
    }
    tmp_counter_ = 0;
    // Keep the complete mapping (2D fields included) for the
    // per-access analysis calls in ConvertBody.
    unit_mapping_ = mapping;

    // For a 2D unit the vectorized coordinate is the selected
    // dimension; the other one stays a serial loop var. The converted
    // body is the innermost compute body.
    const ForNode *inner = nullptr;
    if (const auto *body_for = loop->body.as<ForNode>()) {
      if (body_for->kind == ForKind::kParallel) {
        inner = body_for;
      }
    }
    const Var vector_var =
        mapping.is_2d
            ? (mapping.select_inner ? mapping.inner_var : mapping.outer_var)
            : loop->loop_var;
    loop_var_ = vector_var;
    lanes_ = mapping.lanes;
    // The chunk loop covers the padded chunk count Q = P / L
    // (mapping.chunk_count), which includes the tail chunk of a
    // non-divisible extent.
    nchunks_ = mapping.chunk_count;
    chunk_var_ = Var("q", vector_var.dtype());
    lane_var_ = mapping.lane_var; // analysis placeholder; never escapes
    chunk_analysis_ = mapping.chunk_var;

    // The substitution loop_var -> chunk_analysis_ * L + lane_var_ drives
    // all address/index rewriting (same expression the shared helper
    // uses).
    lane_indexed_expr_ = mapping.index_expr;

    // A divisible extent uses the caller's shared full mask (constant
    // across chunks). A non-divisible extent gets a per-unit scalar
    // `remaining` that starts at the logical extent E and decreases by L
    // per chunk; each chunk binds a dynamic create_mask(remaining, L)
    // before the body and updates remaining at the chunk end. The tail
    // chunk's mask zeroes the lanes past E, which protects the store
    // (loads still read the full padded range — the caller guarantees
    // its readability; the store mask never widens a write past E).
    // The mask var must exist before the body conversion: the
    // converted vstore/vadd reference it while the body is built.
    const bool has_tail = mapping.tail_lanes != 0;
    if (!has_tail) {
      mask_var_ = shared_mask_;
    } else {
      // decl_buffer shapes the Var with a local.var pointer type; the
      // codegen reads the scope from there and emits the scalar
      // local-variable form the PTODSL surface expects.
      remaining_buffer_ =
          tirx::decl_buffer({IntImm(DataType::Int(32), 1)}, DataType::Int(32),
                            "remaining", "local.var");
      // The per-chunk mask bind is emitted inside the chunk loop below;
      // this var is what it binds and what the body references.
      mask_var_ = Var("mask", DataType::Bool(static_cast<int>(lanes_)));
    }

    // Convert the unit body statements in order. For a 2D unit this is
    // the innermost compute body; the outer (kept) coordinate is
    // re-emitted as a serial loop around the chunk loop.
    Array<Stmt> unit_stmts;
    for (const Buffer &old : storage_.PrivateAllocations(loop)) {
      if (old.scope() == "local.var") {
        pto_analysis::CheckBroadcastDType(old->dtype,
                                          "default zero initialization of " +
                                              std::string(old->name));
        PrimExpr zero = BroadcastValue(make_zero(old->dtype), &unit_stmts);
        unit_stmts.push_back(BufferStore(promoted_.at(old->data), zero,
                                         {make_zero(DataType::Int(32))}));
      }
    }
    ConvertBody(inner != nullptr ? static_cast<const Stmt &>(inner->body)
                                 : static_cast<const Stmt &>(loop->body),
                &unit_stmts);

    // Drop binds whose value is never used (address-only aliases and
    // unused source binds strand vci/vbrc/vadd/vload chains) before the
    // chunk loop is built.
    Stmt eliminated = DeadBindElim()(SeqStmt::Flatten(unit_stmts));
    Array<Stmt> chunk_seq;
    if (!has_tail) {
      chunk_seq.push_back(std::move(eliminated));
    } else {
      // Per-chunk mask bind + body + remaining update, in that order.
      // mask_var_ was created before the body conversion.
      PrimExpr remaining_load =
          BufferLoad(remaining_buffer_, {make_zero(DataType::Int(32))});
      PrimExpr mask_value =
          Call(DataType::Bool(static_cast<int>(lanes_)), VmiOp("create_mask"),
               {remaining_load},
               {{"size", IntImm(DataType::Int(32), static_cast<int>(lanes_))}});
      chunk_seq.push_back(Bind(mask_var_, mask_value));
      chunk_seq.push_back(std::move(eliminated));
      PrimExpr next_remaining =
          remaining_load -
          IntImm(DataType::Int(32), static_cast<int64_t>(lanes_));
      chunk_seq.push_back(BufferStore(remaining_buffer_, next_remaining,
                                      {make_zero(DataType::Int(32))}));
    }
    Stmt chunk_body = SeqStmt::Flatten(chunk_seq);
    Stmt chunk_loop =
        For(chunk_var_, make_zero(chunk_var_->dtype),
            IntImm(chunk_var_->dtype, static_cast<int64_t>(nchunks_)),
            ForKind::kSerial, chunk_body);
    if (inner == nullptr) {
      if (!has_tail) {
        return chunk_loop;
      }
      // remaining lives across the whole chunk loop: declare it, set it
      // to E once before the loop.
      Stmt init =
          BufferStore(remaining_buffer_,
                      IntImm(DataType::Int(32),
                             static_cast<int64_t>(mapping.logical_extent)),
                      {make_zero(DataType::Int(32))});
      return SeqStmt::Flatten(AllocBuffer(remaining_buffer_),
                              SeqStmt::Flatten(init, chunk_loop));
    }
    // 2D: keep the non-vectorized coordinate as the outer
    // serial loop; the vectorized dimension becomes the chunk loop.
    const Var kept_var =
        mapping.select_inner ? loop->loop_var : inner->loop_var;
    const int64_t kept_extent =
        mapping.select_inner ? mapping.outer_extent : mapping.inner_extent;
    Stmt kept_body = chunk_loop;
    if (has_tail) {
      // remaining resets to E inside the kept loop: every row starts a
      // fresh recursion, so rows are independent and the tail chunk of
      // each row masks exactly that row's tail lanes.
      Stmt init =
          BufferStore(remaining_buffer_,
                      IntImm(DataType::Int(32),
                             static_cast<int64_t>(mapping.logical_extent)),
                      {make_zero(DataType::Int(32))});
      kept_body = SeqStmt::Flatten(init, chunk_loop);
    }
    Stmt kept_loop =
        For(kept_var, make_zero(kept_var->dtype),
            IntImm(kept_var->dtype, static_cast<int64_t>(kept_extent)),
            ForKind::kSerial, kept_body);
    if (!has_tail) {
      return kept_loop;
    }
    return SeqStmt::Flatten(AllocBuffer(remaining_buffer_), kept_loop);
  }

  // Substitute the analysis chunk var with the real loop var q (and keep
  // the lane placeholder for lane-binding steps).
  PrimExpr BindChunkToQ(const PrimExpr &expr) const {
    class Replacer : public ExprMutator {
    public:
      Replacer(const VarNode *from, const Var &to) : from_(from), to_(to) {}
      PrimExpr VisitExpr_(const VarNode *op) final {
        if (op == from_) {
          return to_;
        }
        return ffi::GetRef<PrimExpr>(op);
      }

    private:
      const VarNode *from_;
      Var to_;
    };
    return Replacer(chunk_analysis_.get(), chunk_var_)(expr);
  }

  // Rewrite Var references inside metadata objects through value_map_
  // Follows containers (Array/Map) structurally; PrimExprs get a
  // var substitution.
  ObjectRef RemapMetadataVars(const ObjectRef &obj) const {
    if (!obj.defined()) {
      return obj;
    }
    if (const auto *expr = obj.as<PrimExprNode>()) {
      class RemapVars : public ExprMutator {
      public:
        RemapVars(const Map<Var, Var> &vmap) : vmap_(vmap) {}
        PrimExpr VisitExpr_(const VarNode *op) final {
          auto it = vmap_.find(GetRef<Var>(op));
          if (it != vmap_.end()) {
            return (*it).second;
          }
          return GetRef<PrimExpr>(op);
        }

      private:
        const Map<Var, Var> &vmap_;
      };
      return RemapVars(value_map_)(GetRef<PrimExpr>(expr));
    }
    // Containers are rejected by Verify (metadata must be a PrimExpr);
    // reaching here with one is an internal error.
    ICHECK(obj.as<PrimExprNode>() == nullptr)
        << "[VectorizeParallelToPTO] internal error: container metadata "
           "passed Verify";
    return obj;
  }

  PrimExpr SubstituteLaneZero(const PrimExpr &expr) const {
    class ZeroLane : public ExprMutator {
    public:
      ZeroLane(const Var &lane) : lane_(lane) {}
      PrimExpr VisitExpr_(const VarNode *op) final {
        if (op == lane_.get()) {
          return make_zero(lane_->dtype);
        }
        return ffi::GetRef<PrimExpr>(op);
      }

    private:
      Var lane_;
    };
    return ZeroLane(lane_var_)(expr);
  }

  // ------------------------------------------------------------------
  // Statement conversion (pre-binds first, then the statement)
  // ------------------------------------------------------------------
  PrimExpr BroadcastValue(const PrimExpr &value, Array<Stmt> *pre) {
    pto_analysis::CheckBroadcastDType(value.dtype(),
                                      "Parallel scalar assignment");
    return EmitBind("broadcast",
                    Call(VectorDType(value.dtype(), lanes_), VmiOp("vbrc"),
                         {value},
                         {{"size", IntImm(DataType::Int(32), lanes_)}}),
                    pre);
  }

  void ConvertBody(const Stmt &body, Array<Stmt> *out) {
    if (const auto *alloc = body.as<AllocBufferNode>()) {
      ICHECK(storage_.Classify(alloc->buffer, current_unit_) ==
             pto::StorageClass::kPrivate);
      return;
    }
    if (const auto *seq = body.as<SeqStmtNode>()) {
      for (const auto &s : seq->seq) {
        ConvertBody(s, out);
      }
      return;
    }
    if (const auto *bind = body.as<BindNode>()) {
      // Whitelist is Verify's job; defensive check here.
      Array<Stmt> pre;
      PrimExpr value = ConvertExpr(bind->value, &pre).value;
      // Pre-binds first, then the Bind itself.
      for (const auto &s : pre) {
        out->push_back(s);
      }
      Var new_var(bind->var->name_hint, value.dtype());
      bind_env_.Set(bind->var, bind->value); // analysis environment
      value_map_.Set(bind->var, new_var);    // value environment
      value_infos_[new_var] = Info(value);
      out->push_back(Bind(new_var, value));
      return;
    }
    if (const auto *store = body.as<BufferStoreNode>()) {
      Array<Stmt> pre;
      PrimExpr value = ConvertExpr(store->value, &pre).value;
      auto kind = storage_.Classify(store->buffer, current_unit_);
      bool accumulator = reducers_.accumulators.count(store->buffer->data);
      if (kind == pto::StorageClass::kPrivate || accumulator) {
        auto slot = storage_.ResolveSlot(store->buffer, store->indices,
                                         &analyzer_, bind_env_);
        Buffer target = accumulator ? vf_buffers_.at(store->buffer->data)
                                    : promoted_.at(store->buffer->data);
        value = ToVmiWidth(value, lanes_, &pre);
        if (accumulator && unit_mapping_.tail_lanes != 0) {
          auto old = EmitBind("previous_partial",
                              BufferLoad(target, {Integer(slot.slot)}), &pre);
          value = EmitBind(
              "preserve_partial",
              Call(target->dtype, VmiOp("vsel"), {mask_var_, value, old}),
              &pre);
        }
        ICHECK(value.dtype() == target->dtype)
            << "PTO private Store dtype/lanes mismatch";
        for (const Stmt &stmt : pre)
          out->push_back(stmt);
        out->push_back(
            BufferStore(target, value, {IntImm(DataType::Int(32), slot.slot)}));
        return;
      }
      ICHECK(kind != pto::StorageClass::kExternal)
          << "PTO external ordinary storage write";
      auto access = pto_analysis::AnalyzeBufferAccess(
          store->buffer, store->indices, /*is_write=*/true, loop_var_,
          CurrentMapping(), &analyzer_, bind_env_);
      if (access.pattern == pto_analysis::AccessPattern::kContinuous ||
          (access.pattern == pto_analysis::AccessPattern::kLaneUniform &&
           unit_mapping_.logical_extent == 1)) {
        value = ToVmiWidth(value, lanes_, &pre);
        Array<PrimExpr> start_idx;
        for (const auto &idx : access.start_indices) {
          start_idx.push_back(
              analyzer_.Simplify(BindChunkToQ(SubstituteLaneZero(idx))));
        }
        PrimExpr ptr =
            Call(DataType::Handle(), tl::access_ptr(),
                 {BufferLoad(store->buffer, start_idx),
                  IntImm(DataType::Int(32), static_cast<int>(lanes_)),
                  IntImm(DataType::Int(32), 2)});
        for (const auto &s : pre) {
          out->push_back(s);
        }
        out->push_back(Evaluate(
            Call(DataType::Void(), VmiOp("vstore"),
                 {value, ptr, IntImm(DataType::Int(32), 0), mask_var_})));
      } else {
        // Verify rejects lane-uniform stores (the current
        // support boundary), so this branch is unreachable. Treat any
        // survivor as an internal error, same style as the wrapper
        // checks above.
        ICHECK(false)
            << "[VectorizeParallelToPTO] internal error: lane-uniform store `"
            << store->buffer->name
            << "` passed Verify; lane-uniform stores must be rejected by "
               "VerifyParallelToPTO";
      }
      return;
    }
    if (const auto *attr = body.as<AttrStmtNode>()) {
      // Verify guarantees the metadata has no lane-dependent semantics
      // (direct or via bind aliases). Rewrite its Var references through
      // value_map_ so already-converted binds are not left dangling
      // The body is converted underneath.
      Array<Stmt> stmts;
      // The body is a nested statement scope — keep the
      // source-bind and value maps scoped with it, so a bind defined
      // inside cannot remap a later sibling's reference (Verify rejects
      // that input; this keeps the transformation state consistent with
      // the unit visitor's scope handling).
      Map<Var, PrimExpr> saved_env = bind_env_;
      Map<Var, Var> saved_map = value_map_;
      ConvertBody(attr->body, &stmts);
      bind_env_ = std::move(saved_env);
      value_map_ = std::move(saved_map);
      Stmt inner = SeqStmt::Flatten(stmts);
      // Default to the *original* node. Only PrimExpr metadata is
      // remapped; a non-object Any (int/str atom metadata) must forward
      // unchanged — the old code wrote back a default-constructed (empty)
      // node when the try_cast failed.
      ffi::Any node = attr->node;
      if (auto ne = attr->node.try_cast<ObjectRef>()) {
        if (const auto *pe = ne.value().as<PrimExprNode>()) {
          node = RemapMetadataVars(GetRef<PrimExpr>(pe));
        }
      }
      PrimExpr value = attr->value;
      if (const auto *ve = attr->value.as<PrimExprNode>()) {
        value = RemapMetadataVars(GetRef<PrimExpr>(ve)).as<PrimExpr>().value();
      }
      out->push_back(AttrStmt(node, attr->attr_key, value, inner));
      return;
    }
    if (const auto *eval = body.as<EvaluateNode>()) {
      // Convert the expression with the same machinery so no stale loop
      // variable survives; the whitelist is Verify's job.
      Array<Stmt> pre;
      PrimExpr value = ConvertExpr(eval->value, &pre).value;
      for (const auto &s : pre) {
        out->push_back(s);
      }
      out->push_back(Evaluate(value));
      return;
    }
    LOG(FATAL) << "[VectorizeParallelToPTO] internal error: unsupported "
                  "statement passed Verify: "
               << body->GetTypeKey();
  }

  pto_analysis::PtoInverseMapping CurrentMapping() const {
    // Return the *complete* inverse mapping saved for this unit.
    // Rebuilding a 1D-only mapping here dropped is_2d, the two loop vars,
    // their extents and select_inner, so the shared access proof lost the
    // kept coordinate's range and Vectorize rejected units Verify had
    // accepted. The analysis placeholders (chunk/lane vars) are exactly
    // the ones saved from InvertPtoLaneLayout; only the emitted chunk
    // loop var q is separate (see BindChunkToQ).
    return unit_mapping_;
  }

  // ------------------------------------------------------------------
  // Expression conversion (index vars resolve through bind_env_)
  // ------------------------------------------------------------------
  struct ConvertedValue {
    PrimExpr value;
    pto::ValueInfo info;
  };

  ConvertedValue ConvertExpr(const PrimExpr &expr, Array<Stmt> *pre) {
    PrimExpr value = ConvertExprValue(expr, pre);
    return {value, Info(value)};
  }

  PrimExpr ConvertExprValue(const PrimExpr &expr, Array<Stmt> *pre) {
    if (const auto *load = expr.as<BufferLoadNode>()) {
      if (vf_buffers_.count(load->buffer->data))
        return ReadRegionValue(expr, pre);
      auto kind = storage_.Classify(load->buffer, current_unit_);
      if (kind == pto::StorageClass::kPrivate ||
          kind == pto::StorageClass::kExternal) {
        auto slot = storage_.ResolveSlot(load->buffer, load->indices,
                                         &analyzer_, bind_env_);
        Buffer buffer = kind == pto::StorageClass::kPrivate
                            ? promoted_.at(load->buffer->data)
                            : storage_.Allocation(load->buffer->data);
        return EmitBind(
            "local_snapshot",
            BufferLoad(buffer, {IntImm(DataType::Int(32), slot.slot)}), pre);
      }
      auto access = pto_analysis::AnalyzeBufferAccess(
          load->buffer, load->indices, /*is_write=*/false, loop_var_,
          CurrentMapping(), &analyzer_, bind_env_);
      if (access.pattern == pto_analysis::AccessPattern::kContinuous ||
          (access.pattern == pto_analysis::AccessPattern::kLaneUniform &&
           unit_mapping_.logical_extent == 1)) {
        Array<PrimExpr> start_idx;
        for (const auto &idx : access.start_indices) {
          start_idx.push_back(
              analyzer_.Simplify(BindChunkToQ(SubstituteLaneZero(idx))));
        }
        PrimExpr ptr =
            Call(DataType::Handle(), tl::access_ptr(),
                 {BufferLoad(load->buffer, start_idx),
                  IntImm(DataType::Int(32), static_cast<int>(lanes_)),
                  IntImm(DataType::Int(32), 1)});
        // VMI with a return value goes through EmitBind.
        return EmitBind("vload",
                        Call(VectorDType(load->dtype, lanes_), VmiOp("vload"),
                             {ptr, IntImm(DataType::Int(32), 0)},
                             {{"size", IntImm(DataType::Int(32),
                                              static_cast<int>(lanes_))}}),
                        pre);
      }
      // lane-uniform scalar load: keep the BufferLoad (lane bound to 0,
      // chunk bound to q — indices resolve through the bind env
      // inside the shared helper already) and bind it to a scalar Var in
      // the current chunk; later uses (vbrc broadcast, scalar arithmetic,
      // store values) reference that Var. Each textual occurrence binds
      // separately — no cross-access load reuse. The Var dtype is the
      // buffer's own scalar dtype.
      Array<PrimExpr> indices;
      for (const auto &idx : access.indices_after_inverse) {
        indices.push_back(
            analyzer_.Simplify(BindChunkToQ(SubstituteLaneZero(idx))));
      }
      return EmitBind("sload", BufferLoad(load->buffer, indices), pre);
    }
    if (const auto *var = expr.as<VarNode>()) {
      auto it = value_map_.find(GetRef<Var>(var));
      if (it != value_map_.end()) {
        return (*it).second;
      }
      if (var == loop_var_.get()) {
        // lane_indexed used as a value: explicit vci vector.
        PrimExpr base = analyzer_.Simplify(
            BindChunkToQ(SubstituteLaneZero(lane_indexed_expr_)));
        return EmitBind(
            "vci",
            Call(VectorDType(var->dtype, lanes_), VmiOp("vci"), {base},
                 {{"size",
                   IntImm(DataType::Int(32), static_cast<int>(lanes_))}}),
            pre);
      }
      return GetRef<PrimExpr>(var);
    }
    if (expr.as<IntImmNode>() || expr.as<FloatImmNode>()) {
      return expr;
    }
    if (const auto *add = expr.as<AddNode>()) {
      return ConvertBinary(add->a, add->b, "vadd", pre);
    }
    if (const auto *sub = expr.as<SubNode>()) {
      return ConvertBinary(sub->a, sub->b, "vsub", pre);
    }
    if (const auto *mul = expr.as<MulNode>()) {
      return ConvertBinary(mul->a, mul->b, "vmul", pre);
    }
    if (const auto *op = expr.as<MaxNode>())
      return ConvertBinary(op->a, op->b, "vmax", pre);
    if (const auto *op = expr.as<MinNode>())
      return ConvertBinary(op->a, op->b, "vmin", pre);
    if (const auto *div = expr.as<DivNode>()) {
      return ConvertBinary(div->a, div->b, "vdiv", pre);
    }
    if (const auto *call = expr.as<CallNode>()) {
      // Hardware-backed unary math (fabs/exp/log/sqrt); the dtype matrix
      // mirrors the ISA documentation (see IsSupportedVectorUnaryOp).
      static const std::unordered_map<std::string, std::string> kUnaryMap = {
          {"tirx.fabs", "vabs"},
          {"tirx.exp", "vexp"},
          {"tirx.log", "vln"},
          {"tirx.sqrt", "vsqrt"}};
      auto call_op = call->op.as<Op>();
      ICHECK(call_op.has_value())
          << "[VectorizeParallelToPTO] internal error: non-Op call passed "
             "Verify: "
          << expr;
      auto it = kUnaryMap.find(call_op.value()->name);
      ICHECK(it != kUnaryMap.end())
          << "[VectorizeParallelToPTO] internal error: call passed Verify "
             "without a VMI mapping: "
          << call_op.value()->name;
      PrimExpr a = ConvertExpr(call->args[0], pre).value;
      if (Info(a).kind == pto::ValueKind::kScalar)
        return Call(expr.dtype(), call->op, {a});
      auto width = Info(a).logical_lanes;
      return EmitBind(it->second,
                      Call(VectorDType(expr.dtype().element_of(), width),
                           VmiOp(it->second),
                           {a, width == 1 ? GetFullMask(1) : mask_var_}),
                      pre);
    }
    if (const auto *cast = expr.as<CastNode>()) {
      // Convert the input exactly once, then plan from the explicit shape of
      // that converted input (the original Cast carries the element types).
      PrimExpr value = ConvertExpr(cast->value, pre).value;
      pto_analysis::CastContext context;
      int64_t cast_lanes = Info(value).logical_lanes;
      if (Info(value).kind == pto::ValueKind::kScalar) {
        context.value_kind = pto_analysis::CastValueKind::kScalar;
        context.logical_lanes = 1;
      } else {
        ICHECK(value.dtype().lanes() == 1 || value.dtype().lanes() == lanes_)
            << "[VectorizeParallelToPTO] internal error: Cast input has "
               "unexpected lanes "
            << value.dtype().lanes() << " (current VL " << lanes_ << ")";
        context.value_kind = pto_analysis::CastValueKind::kVectorVmi;
        context.logical_lanes = cast_lanes;
      }
      ICHECK_EQ(value.dtype().element_of(), cast->value.dtype())
          << "[VectorizeParallelToPTO] internal error: converted Cast input "
             "element type "
          << value.dtype().element_of() << " does not match the original "
          << cast->value.dtype();
      std::string plan_why;
      auto plan = pto_analysis::PlanValueCast(cast, context, &plan_why);
      ICHECK(plan.has_value())
          << "[VectorizeParallelToPTO] internal error: Cast passed Verify "
             "without a conversion plan: "
          << plan_why;
      if (const auto *scalar =
              std::get_if<pto_analysis::ScalarCastPlan>(&plan.value())) {
        ICHECK_EQ(value.dtype().lanes(), 1)
            << "[VectorizeParallelToPTO] internal error: scalar Cast plan "
               "for a vector input";
        if (scalar->action == pto_analysis::ScalarCastAction::kKeepInput) {
          return value;
        }
        // Rebuild the ordinary scalar Cast with the validated annotations and
        // the original span; the scalar lowering itself is unchanged.
        return Cast(cast->dtype, value, scalar->annotations, cast->span);
      }
      const auto &vector = std::get<pto_analysis::VectorCastPlan>(plan.value());
      PrimExpr current = value;
      for (const auto &step : vector.steps) {
        ICHECK_EQ(current.dtype().element_of(), step.from)
            << "[VectorizeParallelToPTO] internal error: Cast step expects "
            << step.from << ", got " << current.dtype().element_of();
        Map<String, ObjectRef> attrs;
        attrs.Set("to_dtype", StringImm(TvmDtypeName(step.to)));
        PrimExpr call;
        if (step.op == pto_analysis::VectorCastOp::kIntegerReinterpret) {
          call = Call(VectorDType(step.to, cast_lanes),
                      VmiOp("vinterpret_cast"), {current}, attrs);
        } else {
          if (!step.saturate.empty()) {
            attrs.Set("saturate", StringImm(step.saturate));
          }
          if (!step.rounding.empty()) {
            attrs.Set("rounding", StringImm(step.rounding));
          }
          call = Call(VectorDType(step.to, cast_lanes), VmiOp("vcvt"),
                      {current}, attrs);
        }
        current = EmitBind("cast", call, pre);
      }
      // Zero steps is the proven identity; a final type check keeps every
      // multi-step chain honest (source convert once, dtype/VL continuous).
      ICHECK_EQ(current.dtype(), VectorDType(cast->dtype, cast_lanes))
          << "[VectorizeParallelToPTO] internal error: Cast plan ended at "
          << current.dtype() << ", expected "
          << VectorDType(cast->dtype, cast_lanes);
      return current;
    }
    if (const auto *sel = expr.as<SelectNode>()) {
      auto condition = ConvertExpr(sel->condition, pre).value;
      auto yes = ConvertExpr(sel->true_value, pre).value;
      auto no = ConvertExpr(sel->false_value, pre).value;
      int64_t width = std::max({Info(yes).logical_lanes, Info(no).logical_lanes,
                                Info(condition).logical_lanes});
      if (Info(condition).kind == pto::ValueKind::kScalar &&
          Info(yes).kind == pto::ValueKind::kScalar &&
          Info(no).kind == pto::ValueKind::kScalar)
        return Select(condition, yes, no);
      if (Info(condition).kind == pto::ValueKind::kScalar)
        condition =
            EmitBind("condition_mask",
                     Call(DataType::Bool(width), VmiOp("create_mask"),
                          {Select(condition, Integer(width), Integer(0))},
                          {{"size", Integer(width)}}),
                     pre);
      yes = ToVmiWidth(yes, width, pre);
      no = ToVmiWidth(no, width, pre);
      return EmitBind("select",
                      Call(yes.dtype(), VmiOp("vsel"), {condition, yes, no}),
                      pre);
    }
#define PTO_CONVERT_REL(Node, Ref, predicate)                                  \
  if (const auto *rel = expr.as<Node>()) {                                     \
    auto a = ConvertExpr(rel->a, pre).value;                                   \
    auto b = ConvertExpr(rel->b, pre).value;                                   \
    if (Info(a).kind == pto::ValueKind::kScalar &&                             \
        Info(b).kind == pto::ValueKind::kScalar)                               \
      return Ref(a, b);                                                        \
    auto width = std::max(Info(a).logical_lanes, Info(b).logical_lanes);       \
    a = ToVmiWidth(a, width, pre);                                             \
    b = ToVmiWidth(b, width, pre);                                             \
    return EmitBind("compare",                                                 \
                    Call(DataType::Bool(width), VmiOp("vcmp"),                 \
                         {a, b, width == 1 ? GetFullMask(1) : mask_var_,       \
                          StringImm(predicate)}),                              \
                    pre);                                                      \
  }
    PTO_CONVERT_REL(LTNode, LT, "lt")
        PTO_CONVERT_REL(LENode, LE, "le") PTO_CONVERT_REL(GTNode, GT, "gt")
            PTO_CONVERT_REL(GENode, GE, "ge") PTO_CONVERT_REL(EQNode, EQ, "eq")
                PTO_CONVERT_REL(NENode, NE, "ne")
#undef PTO_CONVERT_REL
                    LOG(FATAL)
        << "[VectorizeParallelToPTO] internal error: unsupported "
           "expression passed Verify: "
        << expr->GetTypeKey();
  }

  PrimExpr ConvertBinary(const PrimExpr &lhs, const PrimExpr &rhs,
                         const std::string &vmi_name, Array<Stmt> *pre) {
    auto lhs_value = ConvertExpr(lhs, pre);
    auto rhs_value = ConvertExpr(rhs, pre);
    PrimExpr a = lhs_value.value;
    PrimExpr b = rhs_value.value;
    bool a_vec = lhs_value.info.kind == pto::ValueKind::kVmi;
    bool b_vec = rhs_value.info.kind == pto::ValueKind::kVmi;
    if (!a_vec && !b_vec) {
      // Lane-uniform folds stay scalar with the TIR semantics of the node.
      if (vmi_name == "vadd") {
        return a + b;
      }
      if (vmi_name == "vsub") {
        return a - b;
      }
      if (vmi_name == "vmul") {
        return a * b;
      }
      if (vmi_name == "vdiv")
        return div(a, b);
      if (vmi_name == "vmax")
        return Max(a, b);
      if (vmi_name == "vmin")
        return Min(a, b);
      ICHECK(false) << "[VectorizeParallelToPTO] internal error: unknown "
                       "binary op "
                    << vmi_name;
    }
    int64_t width =
        std::max(lhs_value.info.logical_lanes, rhs_value.info.logical_lanes);
    a = ToVmiWidth(a, width, pre);
    b = ToVmiWidth(b, width, pre);
    return EmitBind(vmi_name,
                    Call(a.dtype(), VmiOp(vmi_name),
                         {a, b, width == 1 ? GetFullMask(1) : mask_var_}),
                    pre);
  }

  // TVM dtype name identical to Python str(dtype) (vmi.py ABI):
  // float32 -> "float32", bfloat16 -> "bfloat16", int8 -> "int8", ...
  static std::string TvmDtypeName(DataType dtype) {
    std::ostringstream os;
    if (dtype.is_bfloat16()) {
      return "bfloat16";
    }
    // The FP8 spellings the PTO dtype parser recognizes; both are the TIR
    // dtype names, not the PTO target names.
    if (dtype.is_float8_e4m3fn()) {
      return "float8_e4m3fn";
    }
    if (dtype.is_float8_e5m2()) {
      return "float8_e5m2";
    }
    if (dtype.is_float()) {
      os << "float" << dtype.bits();
    } else if (dtype.is_int()) {
      os << "int" << dtype.bits();
    } else if (dtype.is_uint()) {
      os << "uint" << dtype.bits();
    } else if (dtype.is_bool()) {
      return "bool";
    } else {
      os << dtype;
    }
    return os.str();
  }

  Var EmitBind(const std::string &name, const PrimExpr &value,
               Array<Stmt> *pre) {
    Var result(name + "_" + std::to_string(tmp_counter_++), value.dtype());
    pre->push_back(Bind(result, value));
    if (const auto *call = value.as<CallNode>())
      if (const auto *op = call->op.as<OpNode>();
          op && std::string(op->name).rfind("tl.vmi.", 0) == 0)
        value_infos_[result] = {pto::ValueKind::kVmi, value.dtype().lanes()};
    return result;
  }

  pto::PtoReducerStates reducers_;
  std::unordered_map<Var, Buffer, ObjectPtrHash, ObjectPtrEqual> vf_buffers_;
  pto::ValueInfoMap value_infos_;
  std::map<int64_t, Var> full_masks_;
  Array<Stmt> mask_prelude_;
  pto::ParallelStorageQuery storage_;
  For current_unit_;
  std::unordered_map<Var, Buffer, ObjectPtrHash, ObjectPtrEqual> promoted_;
  arith::Analyzer analyzer_;
  Var shared_mask_; // one full mask per VF, shared by all units

  // Per-unit state.
  Var loop_var_;
  Var chunk_var_;      // real emitted loop var (q)
  Var chunk_analysis_; // analysis placeholder from the inverse mapping
  Var lane_var_;       // analysis placeholder; never escapes
  PrimExpr lane_indexed_expr_;
  /*! The complete inverse mapping of the unit being converted
   * (2D fields included), consumed by CurrentMapping(). */
  pto_analysis::PtoInverseMapping unit_mapping_;
  int64_t lanes_ = 0;
  int64_t nchunks_ = 0;
  Var mask_var_;
  /*! Per-unit scalar buffer holding `remaining` for a non-divisible
   * extent (invalid for divisible units). */
  tirx::Buffer remaining_buffer_{};
  int tmp_counter_ = 0;
  Map<Var, PrimExpr> bind_env_; // analysis env: original var -> def expr
  Map<Var, Var> value_map_;     // value env: original var -> converted var
};

} // namespace

namespace transform {

tvm::transform::Pass VectorizeParallelToPTO() {
  auto pass_func = [](PrimFunc f, const IRModule &m,
                      const tvm::transform::PassContext &ctx) {
    auto target = f->GetAttr<Target>(tvm::attr::kTarget);
    if (!target.defined() || !TargetIsPTO(target.value())) {
      return f;
    }
    return VectorizeParallelToPTOImpl::Substitute(std::move(f));
  };
  return tvm::tirx::transform::CreatePrimFuncPass(
      pass_func, 0, "tl.VectorizeParallelToPTO", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.VectorizeParallelToPTO",
                        VectorizeParallelToPTO);
}

} // namespace transform

} // namespace tl
} // namespace tvm
