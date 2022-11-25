import logging

from antlr4 import *
from antlr4.InputStream import InputStream

from rtamt.parser.stl.StlLexer import StlLexer
from rtamt.parser.stl.StlParser import StlParser
from rtamt.spec.stl.discrete_time.specification_parser import STLSpecificationParser

from rtamt.parser.stl.error.parser_error_listener import STLParserErrorListener
from rtamt.exception.stl.exception import STLParseException

from rtamt.spec.stl.discrete_time.pastifier import STLPastifier

from rtamt.spec.stl.discrete_time.specification import STLDiscreteTimeSpecification
from rtamt.evaluator.stl.online_evaluator import STLOnlineEvaluator
from rtamt.evaluator.stl.offline_evaluator import STLOfflineEvaluator
from rtamt.spec.stl.dense_time.specification_parser import STLDenseTimeSpecificationParser

from rtamt.enumerations.options import *


class STLDenseTimeSpecification(STLDiscreteTimeSpecification):
    """A class used as a container for STL continuous time specifications
       Inherits STLSpecification

    Attributes:

    """
    def __init__(self,semantics=Semantics.STANDARD, language=Language.PYTHON):
        """Constructor for STL Specification"""
        super(STLDenseTimeSpecification, self).__init__(semantics, language)
        self.visitor = STLDenseTimeSpecificationParser(self)
        self.time_interpretation = TimeInterpretation.DENSE


    def parse(self):
        if self.spec is None:
            raise STLParseException('STL specification if empty')

        # Parse the STL spec - ANTLR4 magic

        entire_spec = self.modular_spec + self.spec
        input_stream = InputStream(entire_spec)
        lexer = StlLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = StlParser(stream)
        parser._listeners = [STLParserErrorListener()]
        ctx = parser.specification_file()

        # Create the visitor for the actual spec nodes
        visitor = STLDenseTimeSpecificationParser(self)
        self.top = visitor.visitSpecification_file(ctx)


    def pastify(self):
        # Translate bounded future STL to past STL
        pastifier = STLPastifier()
        self.top.accept(pastifier)
        past = pastifier.pastify(self.top)
        self.top = past

        # evaluate modular sub-specs
        for key in self.var_subspec_dict:
            node = self.var_subspec_dict[key]
            node.accept(pastifier)
            node = pastifier.pastify(node)
            self.var_subspec_dict[key] = node

    # Online
    def update(self, *args, **kargs):
        # list_inputs:
        # example [['p', [[1.1, 2.2], [1.3, 2.5], [1.7, 2]]], ['q', [[1, 2], [1.5, 3.5]]]

        if self.online_evaluator is None:
            # Initialize the online_evaluator
            self.online_evaluator = STLOnlineEvaluator(self)
            self.top.accept(self.online_evaluator)

        for arg in args:
            var_name = arg[0]
            var_object = arg[1]
            self.var_object_dict[var_name] = var_object

        # evaluate modular sub-specs
        for key in self.var_subspec_dict:
            node = self.var_subspec_dict[key]
            out = self.online_evaluator.evaluate(node, [])
            self.var_object_dict[key] = out

        out = self.online_evaluator.evaluate(self.top, [])

        self.var_object_dict = self.var_object_dict.fromkeys(self.var_object_dict, [])


        return out

    # Offline
    def evaluate(self, *args, **kargs):
        if self.offline_evaluator is None:
            # Initialize the online_evaluator
            self.offline_evaluator = STLOfflineEvaluator(self)
            # self.offline_evaluator = STLOfflineEvaluator(self)
            self.top.accept(self.offline_evaluator)

        for arg in args:
            var_name = arg[0]
            var_object = arg[1]
            self.var_object_dict[var_name] = var_object

        # evaluate modular sub-specs
        for key in self.var_subspec_dict:
            node = self.var_subspec_dict[key]
            out = self.offline_evaluator.evaluate(node, [])
            self.var_object_dict[key] = out

        out = self.offline_evaluator.evaluate(self.top, [])
        self.var_object_dict = self.var_object_dict.fromkeys(self.var_object_dict, [])
        return out

    def declare_var(self, var_name, var_type):
        if var_name in self.vars:
            logging.warning(
                'Variable {} was already declared. It is now overriden with the new declaration.'.format(var_name))

        # Associate to variable name 'var' its type 'type'
        self.var_type_dict[var_name] = var_type

        # Add variable name 'var' to the set of variables
        self.add_var(var_name)
        self.free_vars.add(var_name)
        #instance = self.create_var_from_name(var_name)
        #self.var_object_dict[var_name] = instance
        self.var_object_dict[var_name] = []

        # Add the default variable topic to var
        self.var_topic_dict[var_name] = 'rtamt/{}'.format(var_name)

        self.var_io_dict[var_name] = 'output'








