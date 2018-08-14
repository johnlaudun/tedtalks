

```python
import numpy
import networkx as nx
import matplotlib.pyplot as plt

# Open the file:
thresh_JM = numpy.loadtxt("./ThreshJM.csv", delimiter=',')
```


```python
# Create a network from the matrix
thresh = nx.from_numpy_matrix(thresh_JM)
```


```python
pos = nx.spring_layout(thresh)
fig = plt.figure(figsize=(18,18))
nx.draw(G, pos)
plt.savefig('./outputs/descriptions-thresh.png')
```


```python
nx.write_graphml(G, './outputs/thresh_net.gml')
```


```python
print(len(thresh.nodes()), len(thresh.edges()))
```

    2211 978625

