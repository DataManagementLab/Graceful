# Code-Aligned Specification: Semantic Graph Coarsening and Refinement for GRACEFUL

This document keeps the valid idea from the original proposal, but rewrites the parts that did not match the actual GRACEFUL implementation in this repository.

The key correction is that GRACEFUL does not operate on one homogeneous graph `(A, X)`. It constructs a `dgl.DGLHeteroGraph` with typed SQL-plan nodes, predicate nodes, column/table nodes, and UDF nodes. Therefore, the augmentation should be implemented as a typed UDF-subgraph module, not as a generic adjacency-matrix transform over the whole graph.

## Final Verdict

The method is suitable for GRACEFUL if implemented as:

```text
GRACEFUL graph construction
        |
node-type encoding
        |
column-to-UDF prepasses
        |
SemanticGraphAugmentor over UDF nodes only
        |
original GRACEFUL UDF topological message passing
        |
RET-to-query-plan message passing
        |
query-plan message passing
        |
runtime regression
```

This preserves the original GRACEFUL architecture while adding semantic loop/branch context before the standard UDF message passing.

## Existing GRACEFUL Facts

The current DuckDB collator creates a DGL heterograph in:

```text
models/dataset/plan_graph_batching/dd_plan_batching.py
```

The relevant UDF node types are:

```text
INV
COMP
BRANCH
LOOP
LOOPEND
RET
```

The relevant UDF canonical edge types are defined in:

```text
models/zero_shot_models/specific_models/udf_edge_types.py
```

Examples include:

```text
INV -> COMP
COMP -> COMP
COMP -> BRANCH
BRANCH -> COMP
BRANCH -> LOOP
LOOP -> COMP
LOOP -> LOOPEND
LOOPEND -> RET
```

The model currently follows this order:

```python
features = encode_node_types(graph, raw_features)
apply_prepasses()
topological_mp()
apply_post_udf_passes()
runtime_head()
```

The best integration point is after `apply_prepasses()` and before `topological_mp()`.

## Corrected Mathematical View

Instead of one graph:

```text
G = (A, X)
```

use the actual typed graph:

```text
G_h = (V_t, E_r, X_t)
```

where:

- `t` is a node type such as `COMP`, `LOOP`, or `plan0`.
- `r` is an edge relation such as `COMP_LOOP` or `RET_outcol`.
- `X_t` is the raw feature matrix for node type `t`.

After GRACEFUL node-type encoders:

```text
H_t^(0) = Encoder_t(X_t)
```

The augmentor receives:

```python
graph: dgl.DGLHeteroGraph
feat_dict: dict[str, torch.Tensor]
```

and returns:

```python
refined_feat_dict: dict[str, torch.Tensor]
```

For every existing node type:

```python
refined_feat_dict[node_type].shape == feat_dict[node_type].shape
```

The original graph topology and final runtime output interface remain unchanged.

## What the Augmentor Should Modify

Default behavior should refine only UDF node embeddings:

```text
COMP
BRANCH
LOOP
LOOPEND
RET
```

`INV` may be included optionally, but it is often better to leave it unchanged because it represents invocation/input information.

Do not refine these by default:

```text
plan*
logical_pred_*
filter_column
output_column
column
table
```

Reason: SQL-plan message passing already consumes UDF information through `RET`; refining non-UDF nodes would be a much larger architectural change.

## SemanticGraphAugmentor Interface

Implement a module with this shape:

```python
class SemanticGraphAugmentor(torch.nn.Module):
    def forward(
        self,
        graph: dgl.DGLHeteroGraph,
        feat_dict: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], GraphAugmentorDebug | None]:
        ...
```

Required guarantees:

- If disabled, it returns the original `feat_dict`.
- If no UDF nodes exist, it returns the original `feat_dict`.
- If no loop/branch regions exist, it returns the original `feat_dict`.
- It never deletes node types.
- It never changes the tensor shape for existing node types.
- It never changes labels, loss, or runtime regression head.

## Region Extraction

The original proposal correctly focuses on loop and branch regions. The code-aligned version must account for the fact that explicit region membership is not currently stored in the final DGL graph.

Recommended implementation strategy:

1. Extract loop and branch regions while constructing each UDF graph, before or during `udf_to_graph(...)`.
2. Preserve a mapping from original NetworkX UDF nodes to DGL node type and DGL node ID.
3. Store region membership as either DGL metadata edges or a sidecar batch object.

### Region Types

```python
class RegionType(str, Enum):
    LOOP = "loop"
    BRANCH = "branch"
```

### Region Member

```python
@dataclass
class RegionMember:
    nx_node_id: int
    dgl_node_type: str
    dgl_node_id: int
    role: str
    path_index: int | None = None
```

### Semantic Region

```python
@dataclass
class SemanticRegion:
    region_id: int
    region_type: RegionType
    members: list[RegionMember]
    parent_region_id: int | None = None
    child_region_ids: list[int] = field(default_factory=list)
    region_depth: int = 0
    batch_id: int | None = None
    head_node: RegionMember | None = None
    end_node: RegionMember | None = None
    in_rows: float | None = None
    no_iter: float | None = None
    fixed_iter: float | None = None
```

## Loop Regions

The proposal's loop-supernode idea is aligned with the code because the UDF graph already has:

```text
LOOP
LOOPEND
```

and loop features such as:

```text
in_rows_act
in_rows_est
in_rows_deepdb
in_rows_wj
loop_type
fixed_iter
no_iter
loop_part
```

Use the available feature names from the selected featurization, for example:

```text
LOOP_FEATURES
LOOPEND_FEATURES
```

### Loop Membership

For a matched loop head `l_h` and loop end `l_e`, the conceptual membership is:

```python
desc = nx.descendants(udf_graph, l_h)
anc = nx.ancestors(udf_graph, l_e)
members = (desc & anc) | {l_h, l_e}
```

This is acceptable as a first implementation, but must be validated carefully for nested loops.

### Nested Loop Policy

Use this default:

```text
direct_only
```

Meaning:

- A parent loop pools its own direct body nodes.
- A parent loop may consume child loop/branch supernode embeddings.
- A parent loop should not repeatedly pool every fine node inside nested regions.

This avoids double-counting nested computations.

## Branch Regions

The branch-supernode idea is aligned with the code because the UDF graph has:

```text
BRANCH
```

and branch features such as:

```text
in_rows_act
in_rows_est
in_rows_deepdb
cmops
loop_part
```

However, the original proposal assumed reliable true/false path identities and path hit ratios. These are not guaranteed in the current DGL graph.

### Corrected Branch Treatment

Use path-aware branch pooling only if the extractor can recover reliable path membership.

Default branch representation:

```text
branch condition embedding
+ pooled reachable branch-body embedding
+ branch metadata
```

Optional path-aware representation:

```text
branch condition embedding
+ path0 pooled embedding
+ path1 pooled embedding
+ path masks
+ optional normalized path ratios
```

Do not claim that `path0` is true and `path1` is false unless the source CFG explicitly provides that mapping.

### Branch Hit Ratios

Branch hit ratios should be optional.

Preferred sources:

1. Explicit branch cardinality annotations from the UDF graph.
2. Successor `in_rows` divided by branch `in_rows`.
3. Uniform fallback.

Use masks for missing paths:

```python
path_present = [1.0, 0.0]
```

For the first implementation, it is acceptable to omit `p_b^T` and `p_b^F` and use only available encoded branch features.

## Coarsening

The original pooling idea is valid, but it should be typed and UDF-local.

For each region `r`, gather member embeddings from `feat_dict`:

```python
member_embeddings = [
    feat_dict[member.dgl_node_type][member.dgl_node_id]
    for member in region.members
]
```

Then compute:

```text
u_r = Pool({h_i : i in members(r)})
```

Supported pooling modes:

```text
mean
sum
max
weighted_mean
attention
gated_attention
```

Recommended default:

```text
attention
```

Conservative baseline:

```text
mean
```

### Weighted Mean

Weighted mean is useful, but only if reliable weights exist:

```text
w_i = log1p(max(in_rows_i, 0))
```

If no raw `in_rows` metadata exists for a member, fall back to learned attention or mean pooling.

## Supernode Embeddings

Create one supernode embedding per region:

```text
z_r^(0) = phi_type([u_r, boundary_r, metadata_r, child_context_r])
```

where:

- `u_r` is pooled member context.
- `boundary_r` includes loop head/end or branch condition embeddings.
- `metadata_r` includes scalar region metadata.
- `child_context_r` is optional context from nested region supernodes.

All supernode embeddings must have the same hidden dimension as GRACEFUL:

```text
d_model = 128
```

or derive it dynamically from:

```python
next(iter(feat_dict.values())).shape[-1]
```

### Loop Supernode

Code-aligned loop representation:

```text
z_loop = phi_loop([
    h_LOOP,
    h_LOOPEND,
    pooled_body,
    log1p(max(no_iter, 0)),
    fixed_iter,
    log1p(max(in_rows, 0)),
])
```

Notes:

- `no_iter` may be `-1` when unknown, so clamp before `log1p`.
- `in_rows` should use the same cardinality family as the active model configuration when possible.
- If `LOOPEND` is disabled or absent, use a learned missing-loopend token or zeros.

### Branch Supernode

Code-aligned branch representation:

```text
z_branch = phi_branch([
    h_BRANCH,
    pooled_branch_body,
    optional_path0_pool,
    optional_path1_pool,
    path_present_mask,
    log1p(max(in_rows, 0)),
])
```

For the first version, `pooled_branch_body` is enough.

## Coarse Message Passing

The original coarse message-passing idea is valid.

Build a small coarse graph whose nodes are only:

```text
LOOP_SUPER
BRANCH_SUPER
```

Relations should be typed, not a single adjacency matrix.

Recommended coarse relations:

```text
contains_loop
contains_branch
precedes_loop
precedes_branch
```

Optional path-aware relations:

```text
path0_contains_loop
path1_contains_loop
path0_contains_branch
path1_contains_branch
```

Only enable path-aware relations after branch path extraction is validated.

The update can follow the original proposal:

```text
m_r = Aggregate({W_rel z_s : s in N_rel(r)})
z_r' = LayerNorm(z_r + MLP([z_r, m_r]))
```

The coarse graph must never connect regions belonging to different queries in the batch.

## Refinement

The original gated residual refinement is the best default because it is conservative and shape-preserving.

For each fine UDF node `i`, aggregate the supernode context from regions containing it:

```text
c_i = Aggregate({z_r' : i belongs to r})
```

Then refine:

```text
g_i = sigmoid(W_g [h_i, c_i] + b_g)
h_i' = LayerNorm(h_i + g_i * W_c c_i)
```

Recommended default:

```text
gated_residual
```

Supported fusion modes:

```text
none
residual_sum
concat_mlp
gated_residual
```

`none` should make the augmentor a no-op.

## Storage Strategy

Two storage strategies are possible.

### Recommended First Implementation: Sidecar

Use a sidecar object returned by the collator:

```python
GraphAugmentorBatch
```

This avoids modifying DGL heterograph node types while the method is being validated.

Model input should remain backward compatible:

```python
if len(input) == 2:
    graph, features = input
    augmentor_data = None
elif len(input) == 3:
    graph, features, augmentor_data = input
```

### Later Implementation: Heterograph Metadata

Add metadata-only node types:

```text
LOOP_SUPER
BRANCH_SUPER
```

and membership edges:

```text
COMP -> LOOP_SUPER
LOOP -> LOOP_SUPER
LOOPEND -> LOOP_SUPER
BRANCH -> BRANCH_SUPER
COMP -> BRANCH_SUPER
LOOP -> BRANCH_SUPER
LOOPEND -> BRANCH_SUPER
```

Do not add these supernode types to normal GRACEFUL node encoders unless explicitly testing that variant.

## Configuration

Add conservative flags:

```text
--enable_graph_augmentor
--augmentor_position before_udf_mp
--augmentor_storage_mode sidecar
--augmentor_member_pooling attention
--augmentor_fusion gated_residual
--augmentor_region_member_policy direct_only
--augmentor_include_inv false
--augmentor_refine_ret true
--augmentor_use_branch_paths false
--augmentor_use_branch_ratios false
--augmentor_debug false
```

Recommended default behavior:

```text
disabled unless --enable_graph_augmentor is passed
position = before_udf_mp
storage_mode = sidecar
member_pooling = attention
fusion = gated_residual
region_member_policy = direct_only
use_branch_paths = false
use_branch_ratios = false
```

## Training Objective

Keep the original GRACEFUL runtime loss unchanged.

Do not add auxiliary losses in the first implementation.

Optional future auxiliary losses:

```text
region reconstruction loss
region type prediction
loop-cost contrastive loss
branch-path imbalance regularization
```

These should be ablations, not default behavior.

## Backward Compatibility

The implementation must satisfy:

- Existing checkpoints load when the augmentor is disabled.
- Existing training commands work without new flags.
- Existing inference commands work without new flags.
- `test_with_count_edges_msg_aggr` behavior remains unchanged.
- `plans_have_no_udf` behavior remains unchanged.
- `skip_udf` behavior remains unchanged.
- The regression head input shape remains unchanged.

When the augmentor is enabled, checkpoint state dicts will naturally include new augmentor parameters.

## Validation Checklist

Region extraction:

- Every `LOOP` region has a valid head.
- Every `LOOPEND` used by a loop exists if loop-end nodes are enabled.
- Every `BRANCH` region has at least one member.
- Nested regions are acyclic.
- No region crosses query boundaries in a batch.
- All member node IDs are valid for their DGL node type.

Shape safety:

- `refined_feat_dict.keys()` includes all original `feat_dict.keys()`.
- Every refined tensor has the same shape as before.
- No zero-node type crashes.
- No no-UDF batch crashes.

Model behavior:

- Disabled augmentor gives identical outputs to original GRACEFUL.
- Enabled augmentor runs on CPU and GPU.
- Training, validation, inference, and checkpoint save/load work.

Ablations:

- baseline GRACEFUL
- augmentor with mean pooling
- augmentor with attention pooling
- before UDF message passing
- after UDF message passing
- with/without `RET` refinement
- with/without branch path metadata, if reliable

## Updated Complete Pipeline

The corrected full pipeline is:

```text
Input query and UDF
        |
existing GRACEFUL graph construction
        |
DGL heterograph with typed UDF nodes
        |
node-type-specific encoding
        |
column-to-UDF prepasses
        |
SemanticGraphAugmentor
        |-- region extraction metadata
        |-- typed UDF-local coarsening
        |-- loop/branch supernode message passing
        |-- gated residual refinement back to UDF node embeddings
        |
existing GRACEFUL UDF topological message passing
        |
RET-to-plan and query-plan message passing
        |
existing regression head
        |
predicted runtime
```

## Summary of Corrections from the Original Proposal

Keep:

- Loop and branch semantic regions.
- Mean/sum/weighted/attention pooling options.
- Coarse loop/branch supernodes.
- Coarse message passing.
- Refinement back to original nodes.
- Gated residual refinement as a conservative default.
- Original GRACEFUL message passing and runtime head.

Update:

- Replace homogeneous `(A, X)` with typed `DGLHeteroGraph`.
- Apply augmentation to UDF nodes first, not the entire graph.
- Treat `C` as typed membership relations or a sidecar, not a dense global matrix.
- Do not assume true/false branch paths are available.
- Do not assume branch hit ratios are available.
- Use existing loop metadata: `fixed_iter`, `no_iter`, `loop_type`, `loop_part`, and `in_rows_*`.
- Preserve all original tensor shapes and model interfaces.

Final recommendation:

Implement the first version as a disabled-by-default, sidecar-backed `SemanticGraphAugmentor` inserted before GRACEFUL's UDF topological message passing. Use attention pooling and gated residual refinement, but keep branch path ratios and heterograph supernodes as later ablations.
