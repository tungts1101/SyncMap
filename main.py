from FixedChunk import FixedChunk
from SyncMap import SyncMap
from sklearn.cluster import DBSCAN
from sklearn.metrics.cluster import normalized_mutual_info_score

problem = FixedChunk([3, 3, 3, 3], time_delay=10, num_remembered=3)
inputs = problem.get_sequence(4000)
solver = SyncMap(inputs.size(1), dimension=2, adaption_rate=0.012)
solver.run(inputs)
result = solver.get_result()
weights = result.cpu()
labels = DBSCAN(eps=0.3, min_samples=2).fit_predict(weights)
true_labels = problem.get_true_labels()

print("Learned Labels: ",labels)
print("Correct Labels: ",true_labels)
print(normalized_mutual_info_score(labels, true_labels))