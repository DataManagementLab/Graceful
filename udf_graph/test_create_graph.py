import unittest

import networkx
from python_graphs import control_flow
from python_graphs.control_flow import ControlFlowGraph

from udf_graph.create_graph import getUDFgraph


class MyTestCase(unittest.TestCase):

    def cmp_graphs(self, expected_graph, received_graph):
        self.assertEqual(list(expected_graph.nodes), list(received_graph.nodes),
                         f'Nodes do not match ({expected_graph.nodes} vs. {received_graph.nodes})\n{[received_graph.nodes[n]["type"] for n in received_graph.nodes]}')

        for n in expected_graph.nodes:
            for feat_name, val in expected_graph.nodes[n].items():
                self.assertTrue(feat_name in received_graph.nodes[n],
                                f'Node {n} does not have feature {feat_name} ({received_graph.nodes[n]})')
                self.assertEqual(val, received_graph.nodes[n][feat_name],
                                 f'Node {n} feature {feat_name} does not match ({received_graph.nodes[n]}')

        self.assertEqual(expected_graph.edges, received_graph.edges)

    def test_getUDFgraph(self):
        func = '''def func_48(abc:str) -> float:
    if abc != 3:
        i2122 = 0
        while i2122 < 12:
            i2122 += 1
    else:
        abc*3/98
    return abc
        '''
        cf_graph: ControlFlowGraph = control_flow.get_control_flow_graph(func)

        udf_graph, cm = getUDFgraph(cf_graph, func, is_duckdb=True, add_loop_end_node=True)

        expected_graph = networkx.DiGraph()
        expected_graph.add_node(0, type='VAR', var_id='abc')
        expected_graph.add_node(1, type='INVOCATION')
        expected_graph.add_node(2, type='BRANCH', cmops='NotEq', loop_part='False', lineno=2)
        expected_graph.add_node(4, type='LOOP_HEAD', no_iter=12, lineno=4, loop_part='False', fixed_iter='True')
        expected_graph.add_node(5, type='COMP', lineno=5, loop_part='True', ops=['Add'])
        expected_graph.add_node(6, type='LOOP_END', loop_part='False')
        expected_graph.add_node(7, type='COMP', ops=['Div', 'Mult'], loop_part='False', lineno=7,
                                branch_cond='NOT "abc" != 3')
        expected_graph.add_node(8, type='RETURN')
        expected_graph.add_edge(0, 1)
        expected_graph.add_edge(1, 2)
        expected_graph.add_edge(0, 7)
        expected_graph.add_edge(2, 4)
        expected_graph.add_edge(2, 7)
        expected_graph.add_edge(7, 8)
        expected_graph.add_edge(4, 5)
        expected_graph.add_edge(4, 6)
        expected_graph.add_edge(5, 6)
        expected_graph.add_edge(6, 8)

        self.cmp_graphs(expected_graph, udf_graph)

        func2 = '''def func_48(abc:str) -> float:
            if abc != 3:
                i2122 = 0
                while i2122 < 12:
                    i2122 += 1
                    if abc < 3:
                        1+2
                    else:
                        3*4
            else:
                abc*3/98
            return abc
                '''
        cf_graph2: ControlFlowGraph = control_flow.get_control_flow_graph(func2)
        udf_graph2, cm2 = getUDFgraph(cf_graph2, func2, is_duckdb=True, add_loop_end_node=True)

        expected_graph2 = networkx.DiGraph()
        expected_graph2.add_node(0, type='VAR', var_id='abc')
        expected_graph2.add_node(1, type='INVOCATION')
        expected_graph2.add_node(2, type='BRANCH', cmops='NotEq', loop_part='False', lineno=2)
        expected_graph2.add_node(4, type='LOOP_HEAD', no_iter=12, lineno=4, loop_part='False', fixed_iter='True')
        expected_graph2.add_node(5, type='COMP', lineno=5, loop_part='True', ops=['Add'])
        expected_graph2.add_node(6, type='BRANCH', cmops='Lt', loop_part='True', lineno=6)
        expected_graph2.add_node(7, type='COMP', lineno=7, loop_part='True', ops=['Add'])
        expected_graph2.add_node(8, type='COMP', lineno=9, loop_part='True', ops=['Mult'], branch_cond='NOT "abc" < 3')
        expected_graph2.add_node(9, type='LOOP_END', loop_part='False')
        expected_graph2.add_node(10, type='COMP', ops=['Div', 'Mult'], loop_part='False', lineno=11,
                                 branch_cond='NOT "abc" != 3')
        expected_graph2.add_node(11, type='RETURN')
        expected_graph2.add_edge(0, 1)
        expected_graph2.add_edge(0, 10)
        expected_graph2.add_edge(1, 2)

        expected_graph2.add_edge(2, 4)
        expected_graph2.add_edge(4, 5)
        expected_graph2.add_edge(4, 9)
        expected_graph2.add_edge(5, 6)
        expected_graph2.add_edge(6, 7)
        expected_graph2.add_edge(6, 8)
        expected_graph2.add_edge(7, 9)
        expected_graph2.add_edge(8, 9)
        expected_graph2.add_edge(9, 11)

        expected_graph2.add_edge(2, 10)
        expected_graph2.add_edge(10, 11)

        self.cmp_graphs(expected_graph2, udf_graph2)

        cf_graph2: ControlFlowGraph = control_flow.get_control_flow_graph(func2)
        udf_graph2, cm2 = getUDFgraph(cf_graph2, func2, is_duckdb=True, add_loop_end_node=False)

        expected_graph2 = networkx.DiGraph()
        expected_graph2.add_node(0, type='VAR', var_id='abc')
        expected_graph2.add_node(1, type='INVOCATION')
        expected_graph2.add_node(2, type='BRANCH', cmops='NotEq', loop_part='False', lineno=2)
        expected_graph2.add_node(4, type='LOOP_HEAD', no_iter=12, lineno=4, loop_part='False', fixed_iter='True')
        expected_graph2.add_node(5, type='COMP', lineno=5, loop_part='True', ops=['Add'])
        expected_graph2.add_node(6, type='BRANCH', cmops='Lt', loop_part='True', lineno=6)
        expected_graph2.add_node(7, type='COMP', lineno=7, loop_part='True', ops=['Add'])
        expected_graph2.add_node(8, type='COMP', lineno=9, loop_part='True', ops=['Mult'], branch_cond='NOT "abc" < 3')
        expected_graph2.add_node(9, type='COMP', ops=['Div', 'Mult'], loop_part='False', lineno=11,
                                 branch_cond='NOT "abc" != 3')
        expected_graph2.add_node(10, type='RETURN')
        expected_graph2.add_edge(0, 1)
        expected_graph2.add_edge(0, 9)
        expected_graph2.add_edge(1, 2)

        expected_graph2.add_edge(2, 4)
        expected_graph2.add_edge(4, 5)
        expected_graph2.add_edge(5, 6)
        expected_graph2.add_edge(6, 7)
        expected_graph2.add_edge(6, 8)
        expected_graph2.add_edge(7, 10)
        expected_graph2.add_edge(8, 10)
        expected_graph2.add_edge(9, 10)

        expected_graph2.add_edge(2, 9)

        self.cmp_graphs(expected_graph2, udf_graph2)

        func3 = '''def func_45(id:int) -> int:
               	for i3719 in range(12):
               		for i4895 in range(10):
               			var853 = 8+3
               	return id
               '''
        cf_graph3: ControlFlowGraph = control_flow.get_control_flow_graph(func3)
        udf_graph3, cm3 = getUDFgraph(cf_graph3, func3, is_duckdb=True, add_loop_end_node=True)

        expected_graph3 = networkx.DiGraph()
        expected_graph3.add_node(0, type='VAR', var_id='id')
        expected_graph3.add_node(1, type='INVOCATION')
        expected_graph3.add_node(3, type='LOOP_HEAD', no_iter=12, lineno=2, loop_part='False', fixed_iter='True')
        expected_graph3.add_node(5, type='LOOP_HEAD', no_iter=10, lineno=3, loop_part='True', fixed_iter='True')
        expected_graph3.add_node(6, type='COMP', lineno=4, loop_part='True', ops=['Add'])
        expected_graph3.add_node(7, type='LOOP_END', loop_part='True')
        expected_graph3.add_node(8, type='LOOP_END', loop_part='False')
        expected_graph3.add_node(9, type='RETURN')

        expected_graph3.add_edge(0, 1)
        expected_graph3.add_edge(1, 3)
        expected_graph3.add_edge(3, 5)
        expected_graph3.add_edge(3, 8)
        expected_graph3.add_edge(5, 6)
        expected_graph3.add_edge(5, 7)
        expected_graph3.add_edge(6, 7)
        expected_graph3.add_edge(7, 8)
        expected_graph3.add_edge(8, 9)

        self.cmp_graphs(expected_graph3, udf_graph3)

        func4 = '''def func_45(id:int) -> int:
        if id != 3:
            8+3
        else:
            8*3
        7+3
        13-3
        return id
        '''

        cf_graph4: ControlFlowGraph = control_flow.get_control_flow_graph(func4)
        udf_graph4, cm4 = getUDFgraph(cf_graph4, func4, is_duckdb=True, add_loop_end_node=True)

        expected_graph4 = networkx.DiGraph()
        expected_graph4.add_node(0, type='VAR', var_id='id')
        expected_graph4.add_node(1, type='INVOCATION')
        expected_graph4.add_node(2, type='BRANCH', cmops='NotEq', loop_part='False', lineno=2)
        expected_graph4.add_node(3, type='COMP', lineno=3, loop_part='False', ops=['Add'], branch_cond='"id" != 3')
        expected_graph4.add_node(4, type='COMP', lineno=5, loop_part='False', ops=['Mult'], branch_cond='NOT "id" != 3')
        expected_graph4.add_node(5, type='COMP', lineno=6, loop_part='False', ops=['Add'])
        expected_graph4.add_node(6, type='COMP', lineno=7, loop_part='False', ops=['Sub'])
        expected_graph4.add_node(7, type='RETURN')

        expected_graph4.add_edge(0, 1)
        expected_graph4.add_edge(1, 2)
        expected_graph4.add_edge(2, 3)
        expected_graph4.add_edge(2, 4)
        expected_graph4.add_edge(3, 5)
        expected_graph4.add_edge(4, 5)
        expected_graph4.add_edge(5, 6)
        expected_graph4.add_edge(6, 7)

        self.cmp_graphs(expected_graph4, udf_graph4)

        func5 = '''def func_4125(movie_id:int,keyword_id:int) -> int:
        if keyword_id > movie_id:
            var314 = keyword_id-movie_id-45
        else:
            var773 = 35/movie_id*19+keyword_id
        if keyword_id <= movie_id:
            if keyword_id < 858.0:
                3+5
            else:
                65*movie_id*keyword_id*48*91
        else:
            i9122 = 0
            while i9122 < 9:
                i9122 += 1
                var52 = movie_id*36-keyword_id
        return keyword_id
	    '''

        cf_graph5: ControlFlowGraph = control_flow.get_control_flow_graph(func5)
        udf_graph5, cm5 = getUDFgraph(cf_graph5, func5, is_duckdb=True, add_loop_end_node=True)

        expected_graph5 = networkx.DiGraph()
        expected_graph5.add_node(0, type='VAR', var_id='movie_id')
        expected_graph5.add_node(1, type='VAR', var_id='keyword_id')
        expected_graph5.add_node(2, type='INVOCATION')
        expected_graph5.add_node(3, type='BRANCH', cmops='Gt', loop_part='False', lineno=2)
        expected_graph5.add_node(4, type='COMP', lineno=3, loop_part='False', ops=['Sub', 'Sub'],
                                 branch_cond='"keyword_id" > "movie_id"')
        expected_graph5.add_node(5, type='COMP', lineno=5, loop_part='False', ops=['Add', 'Mult', 'Div'],
                                 # not sure why this order - doesnt matter because onehot
                                 branch_cond='NOT "keyword_id" > "movie_id"')

        expected_graph5.add_node(6, type='BRANCH', cmops='LtE', loop_part='False', lineno=6)
        expected_graph5.add_node(7, type='BRANCH', cmops='Lt', loop_part='False', lineno=7)
        expected_graph5.add_node(8, type='COMP', lineno=8, loop_part='False', ops=['Add'])
        expected_graph5.add_node(9, type='COMP', lineno=10, loop_part='False', ops=['Mult', 'Mult', 'Mult', 'Mult'])

        expected_graph5.add_node(11, type='LOOP_HEAD', no_iter=9, lineno=13, loop_part='False', fixed_iter='True')
        expected_graph5.add_node(12, type='COMP', lineno=14, loop_part='True', ops=['Add'])
        expected_graph5.add_node(13, type='COMP', lineno=15, loop_part='True',
                                 ops=['Sub', 'Mult'])  # not sure why this order - doesnt matter because onehot
        expected_graph5.add_node(14, type='LOOP_END', no_iter=9, loop_part='False', fixed_iter='True')
        expected_graph5.add_node(15, type='RETURN')

        expected_graph5.add_edge(0, 2)
        expected_graph5.add_edge(0, 4)
        expected_graph5.add_edge(0, 5)
        expected_graph5.add_edge(0, 9)
        expected_graph5.add_edge(0, 13)

        expected_graph5.add_edge(1, 4)
        expected_graph5.add_edge(1, 5)
        expected_graph5.add_edge(1, 9)
        expected_graph5.add_edge(1, 13)
        expected_graph5.add_edge(1, 2)
        expected_graph5.add_edge(2, 3)
        expected_graph5.add_edge(3, 4)
        expected_graph5.add_edge(3, 5)
        expected_graph5.add_edge(4, 6)
        expected_graph5.add_edge(5, 6)
        expected_graph5.add_edge(6, 7)
        expected_graph5.add_edge(7, 8)
        expected_graph5.add_edge(7, 9)
        expected_graph5.add_edge(8, 15)
        expected_graph5.add_edge(9, 15)
        expected_graph5.add_edge(6, 11)
        expected_graph5.add_edge(11, 12)
        expected_graph5.add_edge(12, 13)
        expected_graph5.add_edge(13, 14)
        expected_graph5.add_edge(14, 15)
        expected_graph5.add_edge(11, 14)

        self.cmp_graphs(expected_graph5, udf_graph5)

        cf_graph5: ControlFlowGraph = control_flow.get_control_flow_graph(func5)
        udf_graph5, cm5 = getUDFgraph(cf_graph5, func5, is_duckdb=True, add_loop_end_node=False)

        expected_graph5 = networkx.DiGraph()
        expected_graph5.add_node(0, type='VAR', var_id='movie_id')
        expected_graph5.add_node(1, type='VAR', var_id='keyword_id')
        expected_graph5.add_node(2, type='INVOCATION')
        expected_graph5.add_node(3, type='BRANCH', cmops='Gt', loop_part='False', lineno=2)
        expected_graph5.add_node(4, type='COMP', lineno=3, loop_part='False', ops=['Sub', 'Sub'],
                                 branch_cond='"keyword_id" > "movie_id"')
        expected_graph5.add_node(5, type='COMP', lineno=5, loop_part='False', ops=['Add', 'Mult', 'Div'],
                                 # not sure why this order - doesnt matter because onehot
                                 branch_cond='NOT "keyword_id" > "movie_id"')

        expected_graph5.add_node(6, type='BRANCH', cmops='LtE', loop_part='False', lineno=6)
        expected_graph5.add_node(7, type='BRANCH', cmops='Lt', loop_part='False', lineno=7)
        expected_graph5.add_node(8, type='COMP', lineno=8, loop_part='False', ops=['Add'])
        expected_graph5.add_node(9, type='COMP', lineno=10, loop_part='False', ops=['Mult', 'Mult', 'Mult', 'Mult'])

        expected_graph5.add_node(11, type='LOOP_HEAD', no_iter=9, lineno=13, loop_part='False', fixed_iter='True')
        expected_graph5.add_node(12, type='COMP', lineno=14, loop_part='True', ops=['Add'])
        expected_graph5.add_node(13, type='COMP', lineno=15, loop_part='True',
                                 ops=['Sub', 'Mult'])  # not sure why this order - doesnt matter because onehot
        expected_graph5.add_node(14, type='RETURN')

        expected_graph5.add_edge(0, 2)
        expected_graph5.add_edge(0, 4)
        expected_graph5.add_edge(0, 5)
        expected_graph5.add_edge(0, 9)
        expected_graph5.add_edge(0, 13)

        expected_graph5.add_edge(1, 4)
        expected_graph5.add_edge(1, 5)
        expected_graph5.add_edge(1, 9)
        expected_graph5.add_edge(1, 13)
        expected_graph5.add_edge(1, 2)
        expected_graph5.add_edge(2, 3)
        expected_graph5.add_edge(3, 4)
        expected_graph5.add_edge(3, 5)
        expected_graph5.add_edge(4, 6)
        expected_graph5.add_edge(5, 6)
        expected_graph5.add_edge(6, 7)
        expected_graph5.add_edge(7, 8)
        expected_graph5.add_edge(7, 9)
        expected_graph5.add_edge(8, 14)
        expected_graph5.add_edge(9, 14)
        expected_graph5.add_edge(6, 11)
        expected_graph5.add_edge(11, 12)
        expected_graph5.add_edge(12, 13)
        expected_graph5.add_edge(13, 14)

        self.cmp_graphs(expected_graph5, udf_graph5)

        func6 = '''def func_45(id:int) -> int:
    i345 = 0
    while i345 < 12:
        i345 += 1
    return id
       '''
        cf_graph6: ControlFlowGraph = control_flow.get_control_flow_graph(func6)
        udf_graph6, cm6 = getUDFgraph(cf_graph6, func6, is_duckdb=True, add_loop_end_node=True)

        expected_graph6 = networkx.DiGraph()
        expected_graph6.add_node(0, type='VAR', var_id='id')
        expected_graph6.add_node(1, type='INVOCATION')
        expected_graph6.add_node(3, type='LOOP_HEAD', no_iter=12, lineno=3, loop_part='False', fixed_iter='True')
        expected_graph6.add_node(4, type='COMP', lineno=4, loop_part='True', ops=['Add'])
        expected_graph6.add_node(5, type='LOOP_END', loop_part='False')
        expected_graph6.add_node(6, type='RETURN')

        expected_graph6.add_edge(0, 1)
        expected_graph6.add_edge(1, 3)
        expected_graph6.add_edge(3, 4)
        expected_graph6.add_edge(4, 5)
        expected_graph6.add_edge(3, 5)
        expected_graph6.add_edge(5, 6)

        self.cmp_graphs(expected_graph6, udf_graph6)

        cf_graph7: ControlFlowGraph = control_flow.get_control_flow_graph(func6)
        udf_graph7, cm7 = getUDFgraph(cf_graph7, func6, is_duckdb=True, add_loop_end_node=False)

        expected_graph7 = networkx.DiGraph()
        expected_graph7.add_node(0, type='VAR', var_id='id')
        expected_graph7.add_node(1, type='INVOCATION')
        expected_graph7.add_node(3, type='LOOP_HEAD', no_iter=12, lineno=3, loop_part='False', fixed_iter='True')
        expected_graph7.add_node(4, type='COMP', lineno=4, loop_part='True', ops=['Add'])
        expected_graph7.add_node(5, type='RETURN')

        expected_graph7.add_edge(0, 1)
        expected_graph7.add_edge(1, 3)
        expected_graph7.add_edge(3, 4)
        expected_graph7.add_edge(4, 5)

        self.cmp_graphs(expected_graph7, udf_graph7)


if __name__ == '__main__':
    unittest.main()
