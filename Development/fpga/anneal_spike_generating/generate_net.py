#! /usr/bin/python2

# It is generator of system verilog code for
# spiking neural network.
# Generaring is based of template files.
# Conventions:
# 1) All verilog-related code is in .template files
# 2) Syntax of .template files:
#   a) Lines containing __PYTHON_PASTE__ mean python-eval-pastes
#      paste should return string,
#      that will be placed to the position of paste.
#   b) Lines between lines containing __PYTHON_EXEC_BEGIN__
#      and __PYTHON_EXEC_END__ are the python-exec-pastes
#      that will be executed by exec during reading .template file
#   c) It's allowed to use variables (transfered as locals to exec and eval)
#      in pastes, but they should be UPPER_CASE
#   d) It's not allowed to declare anything global in pastes.
#      Create fields of TEMPLATE_LOCALS instead.
#   e) other lines of .template files are normal verilog lines

from collections import namedtuple

class SnnGenerator(object):
    PYTHON_PASTE_STR = "__PYTHON_PASTE__"

    PYTHON_PASTE_BEGIN_STR = "__PYTHON_EXEC_BEGIN__"
    PYTHON_PASTE_END_STR = "__PYTHON_EXEC_END__"


    def __init__(self, net_module_name, template_file,
            neuron_template_file=None,
            destination_file=None):
        self.net_module_name = net_module_name
        self.template_file = template_file
        self.neuron_template_file = neuron_template_file
        self.destination_file = destination_file

        # Format of neurons [
        #                    [1,         ["i1", 2]   ],
        #                    [neuron_id, [neuron inputs]],
        #                    ...
        #                   ]

        self.neurons = []


        # Format of level ("dense", (5))
        #                 (type, shape, extra info...)

        self.previous_level = None

        # Number of neurons in previous level
        self.previous_neurons_count = 0

        self.inputs_count = None
        self.outputs_count = None

    def _generate_neuron(self, in_count, template=None, dest=None):
        if dest is None:
            dest = "spiking_neuron_%din.sv" % in_count
        if template is None:
            template = self.neuron_template_file
            if template is None:
                raise ValueError("Neuron template is not set")

        template_locals = {"IN_COUNT": in_count,
                "IN_NUMS": list(range(1, in_count + 1))}
        with open(dest, "w") as d:
            d.write("/* This is generated file. See for template */\n\n")
            for i, line in enumerate(open(template)):
                try:
                    pos = line.find(self.PYTHON_PASTE_STR)
                    if pos == -1:
                        d.write(line)
                        continue
                    pos += len(self.PYTHON_PASTE_STR)
                    d.write(eval(line[pos:], template_locals) + "\n")
                except:
                    print "Unexpected error on line " + str(i) + \
                            " in file " + template
                    raise

    def add_input_layer(self, inputs_count):
        if self.previous_level is not None:
            raise RuntimeError("add_input_layer should be called" +
                    " only once and be first")
        self.inputs_count = inputs_count
        self.previous_level = ("input", (inputs_count,))

    def _get_previous_layer_neurons_count(self):
        return reduce(lambda res, x: res*x, self.previous_level[1], 1)

    def _get_previous_layer_first(self):
        return self.previous_neurons_count + 1 - \
                self._get_previous_layer_neurons_count()

    def _get_layer_first(self):
        return self.previous_neurons_count + 1

    def _get_prev_layers_output(self):
        prev_lf = self._get_previous_layer_first()
        lf = self._get_layer_first()

        if self.previous_level[0] == "input":
            prev_layer_outputs = list(["i" + str(i)
                    for i in range(1, self.inputs_count + 1)])
        else:
            prev_layer_outputs = list(range(prev_lf, lf))

        return prev_layer_outputs



    def add_dense_layer(self, neurons_count):
        if self.previous_level is None:
            raise RuntimeError("add_input_layer should be called at first")
        if self.previous_level[0] == "output":
            raise RuntimeError("do not add layers after call add_output_level")

        lf = self._get_layer_first()

        prev_layer_outputs = self._get_prev_layers_output()

        for i in range(lf, lf + neurons_count):
            self.neurons.append([i, prev_layer_outputs])

        self.previous_level = ("dense", (neurons_count,))
        self.previous_neurons_count += self._get_previous_layer_neurons_count()

    def add_output_level(self, outputs_count):
        if self.previous_level is None:
            raise RuntimeError("add_input_layer should be called at first")
        if self.previous_level[0] == "output":
            raise RuntimeError("do not add layers after call add_output_level")

        lf = self._get_layer_first()

        prev_layer_outputs = self._get_prev_layers_output()

        self.outputs_count = outputs_count

        for i in range(lf, lf + outputs_count * 2):
            self.neurons.append([i, prev_layer_outputs])

        self.previous_level = ("output", (outputs_count,))
        self.previous_neurons_count += self._get_previous_layer_neurons_count()

    def generate_net_code(self, dest=None):
        if dest is None:
            dest = self.destination_file
            if dest is None:
                raise ValueError("Destination file should be set!")

        if self.previous_level is None:
            raise RuntimeError("add_input_layer should be called at first")
        if self.previous_level[0] != "output":
            raise RuntimeError("do not call generate_net " +
                    "before add_output_level")

        class TemplateLocals(dict):
            def __init__(self, dic):
                self.update(dic)
                self.gen_neuron_declaration = None
                self["TEMPLATE_LOCALS"] = self

        template_locals = TemplateLocals({
                "NET_MODULE_NAME": self.net_module_name,
                "NEURONS_COUNT": len(self.neurons),
                "INPUTS_COUNT": self.inputs_count,
                "OUTPUTS_COUNT": self.outputs_count
        })


        with open(dest, "w") as d:
            def generate_neurons():
                gnd = lambda *args: d.write(
                        template_locals.gen_neuron_declaration(*args))

                neurons_input_counts = set()

                for neuron in self.neurons:
                    if len(neuron[1]) not in neurons_input_counts:
                        self._generate_neuron(len(neuron[1]))
                        neurons_input_counts.add(len(neuron[1]))
                    gnd(*neuron)



            template_locals["GENERATE_NEURONS"] = generate_neurons

            d.write("/* This is generated file. See for template */\n\n")
            multiline_paste = None
            for i, line in enumerate(open(self.template_file)):
                try:
                    pos = line.find(self.PYTHON_PASTE_STR)
                    if pos != -1:
                        pos += len(self.PYTHON_PASTE_STR)
                        d.write(eval(line[pos:], template_locals) + "\n")
                    elif line.find(self.PYTHON_PASTE_BEGIN_STR) != -1:
                        assert multiline_paste is None
                        multiline_paste = ""
                    elif line.find(self.PYTHON_PASTE_END_STR) != -1:
                        assert multiline_paste is not None
                        exec multiline_paste in template_locals
                        multiline_paste = None
                    else:
                        if multiline_paste is None:
                            d.write(line)
                        else:
                            multiline_paste += line
                except:
                    print "Unexpected error on line " + str(i + 1) + \
                            " in file " + self.template_file
                    print "Line: " + line
                    if multiline_paste is not None:
                        print "When processing paste... \n" + \
                                multiline_paste + "...end of paste\n"
                    raise




net = SnnGenerator(
        net_module_name="spiking_sum_net",
        destination_file="snn.sv",
        template_file="spiking_neural_network.sv.template",
        neuron_template_file="spiking_neuron.sv.template")

net.add_input_layer(2)

net.add_dense_layer(3)

net.add_dense_layer(1)

net.add_output_level(2)

net.generate_net_code()
