from FixedChunk import FixedChunk
from SyncMap import SyncMap
from sklearn.cluster import DBSCAN
from sklearn.metrics.cluster import normalized_mutual_info_score
from parser import Parser
import timeit
import numpy as np

# problem = FixedChunk([3, 3, 3, 3], time_delay=10, num_remembered=3)
start = timeit.default_timer()
problem = Parser()
inputs = problem.get_sequence(1000)
print(f"Total time generate sequence: {timeit.default_timer() - start}")
solver = SyncMap(inputs.size(1), dimension=2, adaption_rate=0.012)
solver.run(inputs)
result = solver.get_result()
weights = result.cpu()
print(f"Total time processing: {timeit.default_timer() - start}")
np.savez_compressed("weight.npz", key="w")
labels = DBSCAN(eps=0.3, min_samples=2).fit_predict(weights)
print(f"Total time clustering: {timeit.default_timer() - start}")
true_labels = problem.get_true_labels()

print("Learned Labels: ",labels)
print("Correct Labels: ",true_labels)
print(normalized_mutual_info_score(labels, true_labels))
end = timeit.default_timer()
print(f"Total time: {end - start}")