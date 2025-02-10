import re
from typing import List

from cross_db_benchmark.benchmark_tools.generate_workload import Operator, LogicalOperator
from cross_db_benchmark.benchmark_tools.dbms.utils import remove_cast_nesting


class NodeOpNotRecognizedException(Exception):
    def __init__(self, message):
        super().__init__(message)


class FilterParsingError(Exception):
    def __init__(self, message):
        super().__init__(message)


class PredicateNode:
    def __init__(self, text=None, children=[], operator=None, literal=None, literal_feature=None, column=None,
                 udf_name=None, sql: str = None):
        self.text = text
        self.children = children
        self.column = column
        self.operator = operator
        self.literal = literal
        self.literal_feature = literal_feature
        self.udf_name = udf_name
        self.sql = sql

    def __str__(self):
        return self.to_tree_rep(depth=0)

    def __repr__(self):
        return str(self.to_dict())

    def __eq__(self, __o):
        if not isinstance(__o, PredicateNode):
            return False
        return self.children == __o.children and self.operator == __o.operator and self.literal == __o.literal and self.udf_name == __o.udf_name and self.column == __o.column

    def to_dict(self):
        return dict(
            column=self.column,
            operator=str(self.operator),
            literal=self.literal,
            literal_feature=self.literal_feature,
            children=[c.to_dict() for c in self.children],
            udf_name=self.udf_name,
            text=self.text
        )

    def lookup_columns(self, plan, **kwargs):
        if self.column is not None:
            self.column = plan.lookup_column_id(self.column, **kwargs)
        for c in self.children:
            c.lookup_columns(plan, **kwargs)

    def parse_lines_recursively(self, parse_baseline=False, duckdb: bool = False):
        self.parse_lines(parse_baseline=parse_baseline, duckdb=duckdb)
        for c in self.children:
            c.parse_lines_recursively(parse_baseline=parse_baseline, duckdb=duckdb)
        # remove any children that have no literal
        if parse_baseline:
            self.children = [c for c in self.children if
                             c.operator in {LogicalOperator.AND, LogicalOperator.OR,
                                            Operator.IS_NOT_NULL, Operator.IS_NULL}
                             or c.literal is not None]

    def parse_lines(self, parse_baseline=False, duckdb: bool = False):
        keywords = [w.strip() for w in self.text.split(' ') if len(w.strip()) > 0]
        if all([k == 'AND' for k in keywords]):
            self.operator = LogicalOperator.AND
        elif all([k == 'OR' for k in keywords]):
            self.operator = LogicalOperator.OR
        else:
            repr_op = [
                # order matters, since we are testing in this order with includes
                ('= ANY', Operator.IN),
                ('>=', Operator.GEQ),
                ('<=', Operator.LEQ),
                ('<>', Operator.NEQ),
                ('!=', Operator.NEQ),
                ('=', Operator.EQ),
                ('>', Operator.GEQ),
                ('<', Operator.LEQ),
                ('!~~', Operator.NOT_LIKE),
                ('~~', Operator.LIKE),
                ('IS NOT NULL', Operator.IS_NOT_NULL),
                ('IS NULL', Operator.IS_NULL)
            ]
            node_op = None
            literal = None
            column = None
            literal_feature = 0

            text_non_escaped_parts = ''
            escaped = False
            for c in self.text:
                if c == "'":
                    escaped = not escaped
                elif escaped:
                    continue
                else:
                    text_non_escaped_parts += c

            if duckdb and text_non_escaped_parts.count('=') > 1:
                # found multiple = in filter condition. This means that an escaping is missing. DuckDB is sometimes doing that.
                # We need to manually add the escaping
                splits = self.text.split('=', 1)
                self.text = f'{splits[0]}=\'{splits[1]}\''

                text_non_escaped_parts = ''
                escaped = False
                for c in self.text:
                    if c == "'":
                        escaped = not escaped
                    elif escaped:
                        continue
                    else:
                        text_non_escaped_parts += c

            for op_rep, op in repr_op:
                assert duckdb is not None
                if duckdb:
                    split_str = f'{op_rep}'
                else:
                    split_str = f' {op_rep} '
                # self.text = self.text + ' ' # not sure when this is needed

                if split_str in text_non_escaped_parts:
                    assert node_op is None, f"Could not parse: {self.text} / {text_non_escaped_parts}"
                    node_op = op

                    splits = self.text.split(split_str)
                    if len(splits) == 2:
                        literal = splits[1]
                        column = splits[0].strip()
                    else:
                        # comparison operator found multiple times in filter condition
                        column = splits[0].strip()
                        literal = split_str.join(splits[1:]).strip()

                    assert isinstance(column, str)

                    # dirty hack to cope with substring calls in
                    is_substring = self.text.startswith('"substring')
                    if is_substring:
                        self.children[0] = self.children[0].children[0]

                    # current restriction: filters on aggregations (i.e., having clauses) are not encoded using
                    # individual columns
                    agg_ops = {'sum(', 'sum (', 'min(', 'min (', 'max(', 'max (', 'avg(', 'avg (', 'count(', 'count ('}
                    is_having = column.lower() in agg_ops or (len(self.children) == 1
                                                              and self.children[0].text in agg_ops)
                    if is_having:
                        column = None
                        self.children = []
                    else:

                        def recursive_inner(n):
                            # column names can be arbitrarily deep, hence find recursively

                            # check for UDF
                            UDF_regex = re.compile('func_(\d*)')
                            UDF_match = UDF_regex.search(n.text)
                            if UDF_match is not None:
                                udf_name = str(UDF_match.group(0))
                            else:
                                udf_name = None
                            if len(n.children) == 0:
                                return n.text, udf_name

                            # recurse
                            child_text, child_udf = recursive_inner(n.children[0])

                            if udf_name is None:
                                udf_name = child_udf

                            return child_text, udf_name

                        # sometimes column is in parantheses
                        if node_op == Operator.IN:
                            literal = self.children[-1].text
                            if len(self.children) == 2:
                                column = self.children[0].text
                            self.children = []
                        elif len(self.children) == 2:
                            literal = self.children[-1].text
                            column, udf_name = recursive_inner(self)
                            self.children = []
                            if udf_name is not None:
                                self.udf_name = udf_name
                                column = None
                        elif len(self.children) == 1:
                            column, udf_name = recursive_inner(self)
                            if udf_name is not None:
                                self.udf_name = udf_name
                                column = None
                            self.children = []
                        elif len(self.children) == 0:
                            pass
                        else:
                            raise NotImplementedError

                        # column and literal are sometimes swapped
                        type_suffixes = ['::bpchar']
                        if column is not None and any([column.endswith(ts) for ts in type_suffixes]):
                            tmp = literal
                            literal = column
                            column = tmp.strip()

                        # additional features for special operators
                        # number of values for in operator
                        if node_op == Operator.IN:
                            literal_feature = literal.count(',')
                        # number of wildcards for LIKE
                        elif node_op == Operator.LIKE or node_op == Operator.NOT_LIKE:
                            literal_feature = literal.count('%')

                        break

            if parse_baseline:
                if node_op in {Operator.IS_NULL, Operator.IS_NOT_NULL}:
                    literal = None
                elif node_op == Operator.IN:
                    literal = literal.split('::')[0].strip("'").strip("{}")
                    literal = [c.strip('"') for c in literal.split('",')]
                else:
                    if '::text' in literal:
                        literal = literal.split("'::text")[0].strip("'")
                    elif '::bpchar' in literal:
                        literal = literal.split("'::bpchar")[0].strip("'")
                    elif '::date' in literal:
                        literal = literal.split("'::date")[0].strip("'")
                    elif '::time without time zone' in literal:
                        literal = literal.split("'::time")[0].strip("'")
                    elif '::double precision' in literal:
                        literal = float(literal.split("'::double precision")[0].strip("'"))
                    elif '::numeric' in literal:
                        literal = float(literal.split("'::numeric")[0].strip("'"))
                    elif '::integer' in literal:
                        literal = float(literal.split("'::integer")[0].strip("'"))
                    # column comparison. ignored.
                    elif re.match(r"\D\w*\.\D\w*", literal.replace('"', '').replace('\'', '').strip()):
                        literal = None
                    else:
                        try:
                            literal = float(literal.strip())
                        except ValueError:
                            # print(
                            #    f"Could not parse literal {literal} (maybe a join condition? if so, this can be ignored)")
                            literal = None

            if node_op is None:
                raise NodeOpNotRecognizedException(f"Could not parse: {self.text}")

            self.column = column
            if column is not None:
                self.column = tuple(column.split('.'))
                assert len(self.column) <= 2, f"Could not parse: {self.text} / {self.column}"
            self.operator = node_op

            # add escaping to literals if necessary
            assert self.sql is not None
            if literal is not None and literal in self.sql and not literal.startswith('\'') and not literal.startswith(
                    '\"') and not literal.strip() == '':
                if f'\'{literal}\'' in self.sql or f'\"{literal}\"' in self.sql:
                    literal = f'\'{literal}\''

            literal = literal.strip()

            self.literal = literal
            self.literal_feature = literal_feature

    def to_tree_rep(self, depth=0):
        rep_text = '\n' + ''.join(['\t'] * depth)
        rep_text += self.text

        for c in self.children:
            rep_text += c.to_tree_rep(depth=depth + 1)

        return rep_text

    def get_udf_names(self) -> List[str]:
        udf_names = []
        if self.udf_name is not None:
            udf_names.append(self.udf_name)
        for c in self.children:
            udf_names += c.get_udf_names()
        return udf_names


def parse_recursively(filter_cond, offset, _class=PredicateNode, sql: str = None):
    escaped = False

    node_text = ''
    children = []

    between_detected = False
    escaping_missing = False  # in duckdb brackets in literals are sometimes not escaped, i.e. we need to manually add escaping

    def process_between():
        assert ' BETWEEN ' in node_text, f'Could not parse: {node_text}'
        assert ' AND ' in node_text, f'Could not parse: {node_text}'
        column, args = node_text.split(' BETWEEN ')
        args = args.split(' AND ')
        assert len(args) == 2, f'Could not parse: {node_text}'

        filter_children = [_class(f'{column} >= {args[0]}', [], sql=sql), _class(f'{column} <= {args[1]}', [], sql=sql)]
        return _class(' AND ', filter_children, sql=sql), offset

    prev_char = None

    while True:
        if offset >= len(filter_cond):
            if between_detected:
                return process_between()

            if escaping_missing:
                node_text += '\''
                escaping_missing = False

            return _class(node_text, children, sql=sql), offset

        if filter_cond[
            offset] == '(' and not escaped and prev_char not in [
            '=']:  # in duckdb brackets in literals are sometimes not escaped
            child_node, offset = parse_recursively(filter_cond, offset + 1, _class=_class, sql=sql)
            children.append(child_node)
        elif filter_cond[offset] == ')' and not escaped and not escaping_missing:
            if between_detected:
                return process_between()
            return _class(node_text, children, sql=sql), offset
        elif filter_cond[offset] == "'":
            escaped = not escaped
            node_text += "'"
        elif filter_cond[offset:].startswith(
                ' AND ') and not escaped and (node_text.strip() != '' or len(children) > 0) and not between_detected:
            if escaping_missing:
                node_text += '\''
                escaping_missing = False

            children.append(parse_recursively(node_text, 0, _class=_class, sql=sql)[0])
            node_text = ' AND '
            offset += 4
            child_node, offset = parse_recursively(filter_cond, offset + 1, _class=_class, sql=sql)
            children.append(child_node)
        elif filter_cond[offset:].startswith('CAST(') and not escaped and node_text.strip() == '':
            child_node, offset = parse_recursively(filter_cond, offset + 5, _class=_class, sql=sql)
            node_text = child_node.text

            assert ' AS ' in node_text
            node_text = node_text.split(' AS ')[0]
        elif filter_cond[offset:].startswith('constant_or_null(') and not escaped:
            _, offset = parse_recursively(filter_cond, offset + 17, _class=_class, sql=sql)
        elif filter_cond[offset:].startswith(' BETWEEN ') and not escaped and node_text.strip() != '':
            between_detected = True
            node_text += ' BETWEEN '
            offset += 8
        else:
            if filter_cond[offset] == '(' and not escaped:
                node_text += '\''
                escaping_missing = True

            node_text += filter_cond[offset]

            # update previous char
            prev_char = filter_cond[offset]
        offset += 1


def parse_filter(filter_cond_orig, sql: str, parse_baseline=False, duckdb: bool = None):
    filter_cond = remove_cast_nesting(filter_cond_orig)[0]

    parse_tree, _ = parse_recursively(filter_cond, offset=0, sql=sql)
    if len(parse_tree.children) != 1:
        raise FilterParsingError(
            f'Filter structure contains more than one root element (expects 1 since always wrapped in parenthesis): {filter_cond}')

    parse_tree = parse_tree.children[0]
    try:
        parse_tree.parse_lines_recursively(parse_baseline=parse_baseline, duckdb=duckdb)
    except Exception as e:
        print(f'{filter_cond}\n{parse_tree}', flush=True)
        raise e
    if parse_tree.operator not in {LogicalOperator.AND, LogicalOperator.OR, Operator.IS_NOT_NULL, Operator.IS_NULL} \
            and parse_tree.literal is None:
        return None

    # combine nested AND nodes into one
    if parse_tree.operator in LogicalOperator:
        while True:
            found = False
            for child in parse_tree.children:
                # check whether child has same operator than parent node
                if child.operator == parse_tree.operator:
                    # remove child from children and add its children to parent
                    parse_tree.children.remove(child)
                    parse_tree.children += child.children
                    found = True
                    break

            if not found:
                break

    if len(parse_tree.children) == 1 and parse_tree.operator in LogicalOperator:
        parse_tree = parse_tree.children[0]

    # derive udf name from filter condition and append it to parse_tree if the filter condition contains a udf
    # parse_text = filter_cond
    # UDF_regex = re.compile('func_(\d*)')
    # UDF_match = UDF_regex.search(parse_text)
    # if UDF_match is not None:
    #     parse_tree.udf_name = str(UDF_match.group(0))
    #     parse_tree.column = None  # filter compares against a UDF, not against a column

    return parse_tree
