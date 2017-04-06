import fcnn_generator as gen
import simple_fcnn as sfcnn

a = sfcnn.FCNN(1337, [2, 1])
code_generator = gen.FCNNGenerator()
print("Network:")
print(a.get_fpga_network(5))
print("Result:")
print(a.predict([1, 0])[-1])
print("Source:")
print(code_generator.generate_network_module(a.get_fpga_network(5)))