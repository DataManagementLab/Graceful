import math
import re

from cross_db_benchmark.benchmark_tools.generate_workload import Aggregator, ExtendedAggregator, LogicalOperator
from cross_db_benchmark.benchmark_tools.dbms.parse_filter import parse_filter, PredicateNode
from cross_db_benchmark.benchmark_tools.dbms.utils import child_prod

estimated_regex = re.compile(
    '\(cost=(?P<est_startup_cost>\d+.\d+)..(?P<est_cost>\d+.\d+) rows=(?P<est_card>\d+) width=(?P<est_width>\d+)\)')
actual_regex = re.compile('\(actual time=(?P<act_startup_cost>\d+.\d+)..(?P<act_time>\d+.\d+) rows=(?P<act_card>\d+)')
op_name_regex = re.compile('->  ([^"(]+)')
workers_planned_regex = re.compile('Workers Planned: (\d+)')
# filter_columns_regex = re.compile('("\S+"."\S+")')
filter_columns_regex = re.compile('([^\(\)\*\+\-\'\= ]+)')
literal_regex = re.compile('(\'[^\']+\'::[^\'\)]+)')


class PlanOperator(dict):

    def __init__(self, plain_content, sql_query: str, children=None, plan_parameters=None, plan_runtime_ms=0):
        super().__init__()
        self.__dict__ = self
        self.plain_content = plain_content

        self.plan_parameters = plan_parameters if plan_parameters is not None else dict()
        self.children = list(children) if children is not None else []
        self.plan_runtime_ms = plan_runtime_ms
        self.sql_query = sql_query

    def parse_lines(self, alias_dict=None, parse_baseline=False, parse_join_conds=False):
        op_line = self.plain_content[0]

        # parse plan operator name
        op_name_match = op_name_regex.search(op_line)
        assert op_name_match is not None
        op_name = op_name_match.groups()[0]
        for split_word in ["on", "using"]:
            if f" {split_word} " in op_name:
                op_name = op_name.split(f" {split_word} ")[0]
        op_name = op_name.strip()

        # operator table
        if " on " in op_line:
            table_name = op_line.split(" on ")[1].strip()
            table_name_parts = table_name.split(" ")

            table_name = table_name_parts[0].strip('"')

            if table_name.endswith('_pkey'):
                table_name = table_name.replace('_pkey', '')

            if '.' in table_name:
                table_name = table_name.split(".")[1].strip('"')

            if len(table_name_parts) > 1 and alias_dict is not None:
                potential_alias = table_name_parts[1]
                if potential_alias != '' and not potential_alias.startswith('('):
                    alias_dict[potential_alias] = table_name

            if 'Subquery Scan' in op_line:
                alias_dict[table_name] = None
            else:
                self.plan_parameters.update(dict(table=table_name))
        else:
            table_name = None

        self.plan_parameters.update(dict(op_name=op_name))

        # parse estimated plan costs
        match_est = estimated_regex.search(op_line)
        assert match_est is not None
        self.plan_parameters.update({k: float(v) for k, v in match_est.groupdict().items()})

        # parse actual plan costs
        match_act = actual_regex.search(op_line)
        if match_act is not None:
            self.plan_parameters.update({k: float(v) for k, v in match_act.groupdict().items()})

        # collect additional optional information
        for l in self.plain_content[1:]:
            l = l.strip()
            workers_planned_match = workers_planned_regex.search(l)

            if workers_planned_match is not None:
                workers_planned = workers_planned_match.groups()
                if isinstance(workers_planned, list) or isinstance(workers_planned, tuple):
                    workers_planned = workers_planned[0]
                workers_planned = int(workers_planned)
                self.plan_parameters.update(dict(workers_planned=workers_planned))

            # Output columns
            # probably in the future it would be easier to define a grammar and use a parser for this
            elif l.startswith("Output: "):
                l = l.replace("Output: ", "")
                output_columns, udf_table_name, udf_params = self.parse_output_columns(l)

                if udf_table_name == 'missing_table_prefix':
                    assert table_name is not None, self.plain_content[0]
                    udf_table_name = table_name
                elif udf_table_name is not None:
                    assert table_name == udf_table_name, f'{table_name} != {udf_table_name} in {l}\n{self.plain_content}'

                self.plan_parameters.update(dict(output_columns=output_columns))
                self.plan_parameters.update(dict(udf_table=udf_table_name))
                self.plan_parameters.update(dict(udf_params=udf_params))

            # Filter columns
            elif l.startswith("Filter: "):
                l = l.replace("Filter: ", "")
                parse_tree = parse_filter(l, parse_baseline=parse_baseline, sql=self.sql_query)
                self.add_filter(parse_tree)

                # Important note: this function is called for both, verbose and analyze output
                # the analyze output does not contain the table name of the udf that is used in the filter
                # hence we have to decide first if we are interested in the output or not
                UDF_regex = re.compile('func_((\d*)\([\w\W,\s]*?\))')
                UDF_match = UDF_regex.search(l)
                if UDF_match is not None:
                    # get the table name of the udf that is used in the filter
                    param_regex = re.compile(r'\([^)]*\)')
                    param_str = param_regex.search(UDF_match.group(0)).group(0)[
                                1:-1]  # [1:-1] removes the brackets at the beginning and end

                    # to determine if a the verbose output is used, we can check if the string contains a dot
                    # as structure  is "table_name"."column_name"
                    if "." in param_str:
                        table_name = param_str.split(',')[0].split(".")[0]  # structure "table_name"."column_name"
                        if table_name[0] == "\"":
                            table_name = table_name[1:-1]
                        self.plan_parameters.update(dict(udf_table=table_name))

                        # get the columns of the udf that are used in the filter
                        param_outlst = []
                        param_str = param_regex.search(UDF_match.group(0)).group(0)[
                                    1:-1]  # [1:-1] removes the brackets at the beginning and end
                        param_lst = param_str.split(',')
                        for elem in param_lst:
                            # get the parameter name
                            param_name = elem.split(".")[1]
                            if param_name[0] == "\"":
                                param_name = param_name[1:-1]
                            param_outlst.append(param_name)
                        self.plan_parameters.update(dict(udf_params=param_outlst))

            # Join Filter
            elif parse_join_conds and l.startswith("Join Filter: "):
                l = l.replace("Join Filter: ", "")
                parse_tree = parse_filter(l, parse_baseline=False, sql=self.sql_query)
                self.add_filter(parse_tree)

            # Hash Condition
            elif parse_join_conds and l.startswith("Hash Cond: "):
                l = l.replace("Hash Cond: ", "")
                parse_tree = parse_filter(l, parse_baseline=False, sql=self.sql_query)
                self.add_filter(parse_tree)

            # Filter columns
            # There were lots of errors during training with index conditions that are concatenated by using AND
            # Hence, as a quick fix, I decided to disregard them. Anyway, there are only very few cases where index
            # conditions are connected with AND
            elif l.startswith("Index Cond: ") and "AND" not in l:
                l = l.replace("Index Cond: ", "")
                parse_tree = parse_filter(l, parse_baseline=parse_baseline, sql=self.sql_query)
                self.add_filter(parse_tree)

            elif l == 'Inner Unique: true':
                self.plan_parameters.update(dict(inner_unique=True))
        self.plain_content = []

    def parse_output_columns(self, l):

        table_name = None
        param_outlst = []
        # If output contains a function call, we will remove the function arguments from the UDF
        # as this information is (a) not needed here and (b) causes problems when splitting the output columns by comma
        UDF_regex = re.compile('func_((\d*)\([\w\W,\s]*?\))')
        UDF_match = UDF_regex.search(l)
        if UDF_match is not None:

            # get the table name where the UDF is applied on
            param_regex = re.compile(r'\([^)]*\)')
            param_str = param_regex.search(UDF_match.group(0)).group(0)[
                        1:-1]  # [1:-1] removes the brackets at the beginning and end

            # check if the string contains a dot as structure  is "table_name"."column_name"
            udf_input_cols = param_str.split(',')
            assert len(udf_input_cols) > 0, f"Could not parse {param_str}"
            table_names = []
            for table_col_str in udf_input_cols:
                table_col_str = table_col_str.strip()
                if "." in table_col_str:
                    table, col = table_col_str.split(".")

                    # remove escaping
                    if table[0] == '\"':
                        table = table[1:-1]

                else:
                    # column name is missing table. prefix
                    table = 'missing_table_prefix'
                    col = table_col_str

                table_names.append(table)

                assert not col[0] == ' ', f'Column name starts with space: |{col}|'
                if col[0] == '\"':
                    col = col[1:-1]
                param_outlst.append(col)

            assert len(set(table_names)) == 1, f"UDF input columns from different tables: {table_names}"
            table_name = table_names[0]

            func_call_no_params = re.sub(r'\([^)]*\)', '', UDF_match.group(0))
            # adjust the function call
            l = re.sub(UDF_regex, func_call_no_params + "()", l)

        output_columns = []
        for col in l.split(', '):
            UDF_regex = re.compile('func_((\d*)\([\w\W,\s]*?\))')
            UDF_match = UDF_regex.search(col)

            # do normal processing if no UDF occurs in Output
            if UDF_match is None:
                # argument in a function call not an actual column
                if col.strip(')').isnumeric() or col in {'NULL::numeric', 'NULL::bigint', 'NULL::integer',
                                                         'NULL::double precision', 'NULL::text', 'NULL::bpchar'}:
                    continue

                columns = []
                if 'count(*)' in col:
                    agg = Aggregator.COUNT
                # for operators to change dates
                elif 'date_part(' in col:
                    continue
                else:
                    # remove literals
                    col = re.sub(literal_regex, '', col)

                    # timestamp types can be disregarded
                    ts_text = '::timestamp without time zone'
                    if ts_text in col:
                        col = col.replace(ts_text, '')

                    assert filter_columns_regex.search(col) is not None, f"Could not parse {col}"

                    for filter_m in filter_columns_regex.finditer(col):
                        curr_col = filter_m.group()
                        endpos = filter_m.span()[-1]

                        # check whether it is just an aggregation
                        if curr_col.lower() in ['avg', 'sum', 'count', 'min', 'max']:
                            # next character should be opening bracket
                            if len(col) > endpos and col[endpos] == '(':
                                continue

                        # additional PG keywords or operators
                        if curr_col.lower() in {'partial', 'precision', 'case', 'when', 'then', 'if', 'else', 'end',
                                                'or', '<>', '/', '~~', 'and', '"substring"', 'distinct'}:
                            continue
                        # just a type
                        if curr_col.startswith('::'):
                            continue
                        # just a literal
                        if curr_col.startswith("'") and curr_col.split("'")[-1].startswith("::") \
                                or curr_col.replace('.', '').isnumeric():
                            continue

                        columns.append(tuple(curr_col.split('.')))

                    assert len(columns) > 0, f"Could not parse {col}"

                    # if there is an aggregation, find it
                    agg = None
                    if 'PARTIAL' in col:
                        col = col.replace('PARTIAL ', '').strip('( ')
                    col = col.strip('(')
                    for curr_agg in list(Aggregator) + list(ExtendedAggregator):
                        if col.startswith(str(curr_agg).lower()):
                            agg = str(curr_agg)

                    # assert agg is not None, f"Could not parse {col}"
                output_columns.append(
                    dict(aggregation=str(agg), columns=columns, udf_name=None, udf_output="False"))
            else:
                # obtain the udf_name => we need this later for the graph creation
                udf_name = str(UDF_match.group(0)).split("(")[0]

                if 'count(*)' in col:
                    agg = Aggregator.COUNT
                else:
                    # if there is an aggregation, find it
                    agg = None
                    if 'PARTIAL' in col:
                        col = col.replace('PARTIAL ', '').strip('( ')
                    col = col.strip('(')
                    for curr_agg in list(Aggregator) + list(ExtendedAggregator):
                        if col.startswith(str(curr_agg).lower()):
                            agg = str(curr_agg)
                output_columns.append(
                    dict(aggregation=str(agg), columns=None, udf_name=udf_name, udf_output="True"))

            return output_columns, table_name, param_outlst

    def add_filter(self, parse_tree):
        if parse_tree is not None:
            existing_filter = self.plan_parameters.get('filter_columns')
            if existing_filter is not None:
                parse_tree = PredicateNode(None, [existing_filter, parse_tree], sql=self.sql_query)
                parse_tree.operator = LogicalOperator.AND

            self.plan_parameters.update(dict(filter_columns=parse_tree))

    def parse_columns_bottom_up(self, column_id_mapping, partial_column_name_mapping, table_id_mapping,
                                alias_dict):

        if alias_dict is None:
            alias_dict = dict()

        # first keep track which tables are actually considered here
        node_tables = set()
        if self.plan_parameters.get('table') is not None:
            node_tables.add(self.plan_parameters.get('table'))

        for c in self.children:
            node_tables.update(
                c.parse_columns_bottom_up(column_id_mapping, partial_column_name_mapping, table_id_mapping, alias_dict))

        self.plan_parameters['act_children_card'] = child_prod(self, 'act_card')
        self.plan_parameters['est_children_card'] = child_prod(self, 'est_card')

        output_columns = self.plan_parameters.get('output_columns')
        if output_columns is not None:
            for output_column in output_columns:
                col_ids = []
                # in case of a UDF, output_column['columns'] is None
                # in all other cases, we will have actual columns
                if output_column['columns'] is not None:
                    for c in output_column['columns']:
                        try:
                            c_id = self.lookup_column_id(c, column_id_mapping, node_tables, partial_column_name_mapping,
                                                         alias_dict)
                            col_ids.append(c_id)
                        except Exception as e:
                            print(e)
                            print(c)
                            print(column_id_mapping)
                            raise e
                            # not c[1].startswith('agg_')
                            if c[0] != 'subgb':
                                raise ValueError(f"Did not find unique table for column {c}")

                    output_column['columns'] = col_ids

        filter_columns = self.plan_parameters.get('filter_columns')
        if filter_columns is not None:
            filter_columns.lookup_columns(self, column_id_mapping=column_id_mapping, node_tables=node_tables,
                                          partial_column_name_mapping=partial_column_name_mapping,
                                          alias_dict=alias_dict)
            self.plan_parameters['filter_columns'] = filter_columns.to_dict()

        # replace table by id
        table = self.plan_parameters.get('table')
        if table is not None:
            if table in table_id_mapping:
                self.plan_parameters['table'] = table_id_mapping[table]
            else:
                del self.plan_parameters['table']

        return node_tables

    def lookup_column_id(self, c, column_id_mapping, node_tables, partial_column_name_mapping, alias_dict):
        assert isinstance(c, tuple)
        # here it is clear which column is meant
        if len(c) == 2:
            table = c[0].strip('"')
            column = c[1].strip('"')

            if table in alias_dict:
                table = alias_dict[table]

                # this is a subquery and we cannot uniquely identify the corresponding table
                if table is None:
                    return self.lookup_column_id((c[1],), column_id_mapping, node_tables, partial_column_name_mapping,
                                                 alias_dict)

        # we now have to guess which table this column belongs to
        elif len(c) == 1:
            column = c[0].strip('"')

            potential_tables = partial_column_name_mapping[column].intersection(node_tables)
            assert len(potential_tables) == 1, f"Did not find unique table for column {column} " \
                                               f"(node_tables: {node_tables})"
            table = list(potential_tables)[0]
        else:
            raise Exception(f"Cannot handle column {c}")

        col_id = column_id_mapping[(table, column)]

        assert isinstance(col_id, int), col_id
        # assert len(col_id)<=2,col_id
        return col_id

    def merge_recursively(self, node):
        assert self.plan_parameters['op_name'] == node.plan_parameters['op_name']
        assert len(self.children) == len(node.children)

        self.plan_parameters.update(node.plan_parameters)
        for self_c, c in zip(self.children, node.children):
            self_c.merge_recursively(c)

    def parse_lines_recursively(self, alias_dict=None, parse_baseline=False, parse_join_conds=False,
                                duckdb: bool = None):
        self.parse_lines(alias_dict=alias_dict, parse_baseline=parse_baseline, parse_join_conds=parse_join_conds)
        for c in self.children:
            c.parse_lines_recursively(alias_dict=alias_dict, parse_baseline=parse_baseline,
                                      parse_join_conds=parse_join_conds, duckdb=duckdb)

    def min_card(self):
        act_card = self.plan_parameters.get('act_card')
        if act_card is None:
            act_card = math.inf

        for c in self.children:
            child_min_card = c.min_card()
            if child_min_card < act_card:
                act_card = child_min_card

        return act_card

    def recursive_str(self, pre):
        pre_whitespaces = ''.join(['\t' for _ in range(pre)])
        # current_string = '\n'.join([pre_whitespaces + content for content in self.plain_content])
        current_string = pre_whitespaces + str(self.plan_parameters)
        node_strings = [current_string]

        for c in self.children:
            node_strings += c.recursive_str(pre + 1)

        return node_strings

    def __str__(self):
        rec_str = self.recursive_str(0)
        return '\n'.join(rec_str)
