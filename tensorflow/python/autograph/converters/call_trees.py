# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Handles function calls, by generating compiled function names and calls.

Note: this transformer does not rename the top level object being converted;
that is the caller's responsibility.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gast

from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import ast_util
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import templates


# TODO(mdan): Rename to FunctionCallsTransformer.


class CallTreeTransformer(converter.Base):
  """Transforms the call tree by renaming transformed symbols."""

  def visit_FunctionDef(self, node):
    node.args = self.visit(node.args)
    node.body = self.visit_block(node.body)
    # TODO(mdan): Is this correct for local functions?
    node.decorator_list = []
    if node.returns:
      node.returns = self.visit(node.returns)
    return node

  def visit_With(self, node):
    # Context manager calls (in node.items) are not converted.
    node.body = self.visit_block(node.body)
    return node

  def visit_Call(self, node):
    # TODO(mdan): Refactor converted_call as a 'Call' operator.

    # Calls to the internal 'ag__' module are never converted (though their
    # arguments might be).
    full_name = str(anno.getanno(node.func, anno.Basic.QN, default=''))
    if full_name.startswith('ag__.'):
      return self.generic_visit(node)
    if (full_name == 'print' and
        not self.ctx.program.options.uses(converter.Feature.BUILTIN_FUNCTIONS)):
      return self.generic_visit(node)

    if isinstance(node.func, gast.Attribute):
      func = gast.Str(node.func.attr)
      owner = node.func.value
    else:
      func = node.func
      owner = parser.parse_expression('None')

    starred_arg = None
    normal_args = []
    for a in node.args:
      if isinstance(a, gast.Starred):
        assert starred_arg is None, 'Multiple *args should be impossible.'
        starred_arg = a
      else:
        normal_args.append(a)
    if starred_arg is None:
      args = templates.replace_as_expression('(args,)', args=normal_args)
    else:
      args = templates.replace_as_expression(
          '(args,) + tuple(stararg)',
          stararg=starred_arg.value,
          args=normal_args)

    kwargs_arg = None
    normal_keywords = []
    for k in node.keywords:
      if k.arg is None:
        assert kwargs_arg is None, 'Multiple **kwargs should be impossible.'
        kwargs_arg = k
      else:
        normal_keywords.append(k)
    if kwargs_arg is None:
      kwargs = ast_util.keywords_to_dict(normal_keywords)
    else:
      kwargs = templates.replace_as_expression(
          'dict(kwargs, **keywords)',
          kwargs=kwargs_arg.value,
          keywords=ast_util.keywords_to_dict(normal_keywords))

    template = """
      ag__.converted_call(func, owner, options, args, kwargs)
    """
    new_call = templates.replace_as_expression(
        template,
        func=func,
        owner=owner,
        options=self.ctx.program.options.to_ast(
            self.ctx,
            internal_convert_user_code=self.ctx.program.options.recursive),
        args=args,
        kwargs=kwargs)

    return new_call


def transform(node, ctx):
  """Transform function call to the compiled counterparts.

  Args:
    node: AST
    ctx: EntityContext
  Returns:
    A tuple (node, new_names):
        node: The transformed AST
        new_names: set(string), containing any newly-generated names
  """
  return CallTreeTransformer(ctx).visit(node)
