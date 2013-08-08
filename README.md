## Testing Functions

I'm honestly being a bit lazy here, because it is a bit annoying to test
nondeterministic/probabilistic functions. But running these routines will help
increase confidence in the correctness of the code.

### TrustGraph#initialize\_agent\_types

```python
import matplotlib as plt
from trust_graph import TrustGraph

# Print out histograms that should look like the underlying distribution
plt.hist(TrustGraph.initialize_agent_types(10000, 'uniform'), bins=20)
plt.hist(TrustGraph.initialize_agent_types(10000, 'normal'), bins=20)
plt.hist(TrustGraph.initialize_agent_types(10000, 'beta'), bins=20)
```

### TrustGraph#initialize\_edges

```python
import matplotlib as plt
import numpy as np
from trust_graph import TrustGraph

at = sorted(TrustGraph.initialize_agent_types(50, 'normal'))

# Should print out a line graph that has a flat trendline
edges = TrustGraph.initialize_edges(at, 'uniform', 20)
plt.plot(np.mean(edges, 1))

# Should print out a line graph that has an increasing trendline
edges = TrustGraph.initialize_edges(at, 'cluster', 20)
plt.plot(np.mean(edges, 1))
```

### TrustGraph#initialize\_edge\_weights

```python
import matplotlib as plt
import numpy as np
from scipy import stats
from trust_graph import TrustGraph

at = sorted(TrustGraph.initialize_agent_types(50, 'normal'))
edges = TrustGraph.initialize_edges(at, 'uniform', 20)
weights = TrustGraph.initialize_edge_weights(at, edges, 'sample', 'normal', 10)

# Verify that the right number of weights were set in the adjacency matrix
(np.array([len(filter(lambda x: x is not None, x)) for x in weights]) == 20).all()  # => True

# Verify that the sampled weights look close to the actual agent types
# The two lines should look pretty similar
plt.plot(stats.stats.nanmean(weights.astype(float), 0)); plt.plot(at);

# Verify that 'noisy' has a bias toward 0.5
weights = TrustGraph.initialize_edge_weights(at, edges, 'noisy', 'normal', 10)
plt.plot(stats.stats.nanmean(weights.astype(float), 0)); plt.plot(at);

# You can look at the other weights, but I'm not sure what kind of visual
# verification would be most effective, since they all have roughly similar
# expected values
weights = TrustGraph.initialize_edge_weights(at, edges, 'prior', 'uniform', 10)
plt.plot(stats.stats.nanmean(weights.astype(float), 0)); plt.plot(at);
weights = TrustGraph.initialize_edge_weights(at, edges, 'prior', 'normal', 10)
plt.plot(stats.stats.nanmean(weights.astype(float), 0)); plt.plot(at);
weights = TrustGraph.initialize_edge_weights(at, edges, 'prior', 'beta', 10)
plt.plot(stats.stats.nanmean(weights.astype(float), 0)); plt.plot(at);
```

### utils.softmax\_rv

TODO
