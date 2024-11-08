udf_node_types = ['COMP', 'INV', 'RET', 'BRANCH', 'LOOP', 'LOOPEND']

udf_canonical_edge_types = [
    ('INV', 'INV_COMP', 'COMP'),
    ('INV', 'INV_BRANCH', 'BRANCH'),
    ('INV', 'INV_LOOP', 'LOOP'),
    ('INV', 'INV_RET', 'RET'),
    ('COMP', 'COMP_COMP', 'COMP'),
    ('COMP', 'COMP_BRANCH', 'BRANCH'),
    ('COMP', 'COMP_LOOP', 'LOOP'),
    ('COMP', 'COMP_RET', 'RET'),
    ('BRANCH', 'BRANCH_BRANCH', 'BRANCH'),
    ('BRANCH', 'BRANCH_COMP', 'COMP'),
    ('BRANCH', 'BRANCH_LOOP', 'LOOP'),
    ('LOOP', 'LOOP_BRANCH', 'BRANCH'),
    ('LOOP', 'LOOP_COMP', 'COMP'),
    ('LOOP', 'LOOP_LOOP', 'LOOP'),
    ('LOOP', 'LOOP_RET', 'RET'),
    ('LOOP', 'LOOP_LOOPEND', 'LOOPEND'),
    ('COMP', 'COMP_LOOPEND', 'LOOPEND'),
    ('LOOPEND', 'LOOPEND_RET', 'RET'),
    ('LOOPEND', 'LOOPEND_COMP', 'COMP'),
    ('LOOPEND', 'LOOPEND_BRANCH', 'BRANCH'),
    ('LOOPEND', 'LOOPEND_LOOP', 'LOOP'),
    ('LOOPEND', 'LOOPEND_LOOPEND', 'LOOPEND')
]
