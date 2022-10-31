from FixedChunk import FixedChunk
from SyncMap import SyncMap

problem = FixedChunk([3, 3, 3], 3, 2)
inputs, labels = problem.get_sequence(10)
solver = SyncMap(len(inputs[0]), 2)
solver.run(inputs)
result = solver.get_result()

print(result)