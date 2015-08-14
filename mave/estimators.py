import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.neighbors.kde import KernelDensity

def normalize(x):
  # normalize numpy array to [0, 1]
  mi = np.min(x) 
  x -= np.sign(mi) * np.abs(mi)
  x /= np.max(x)
  return x

class HourWeekdayBinModel(DummyRegressor):

  def __init__(self, strategy='mean'):
    self.strategy = strategy

  def fit(self, X, y):
    a = np.zeros((24, 7))
    hours = np.copy(X[:, 1])
    weekdays = np.copy(X[:, 2])
    hours = 23 * normalize(hours)
    weekdays = 6 * normalize(weekdays)

    if self.strategy == 'mean':
      counts = a.copy()
      for i, row in enumerate(zip(hours, weekdays)):
        hour = int(row[0])
        day = int(row[1])
        counts[hour, day] += 1
        a[hour, day] += y[i]

      counts[counts == 0] = 1
      self._model = a / counts

    elif self.strategy in ('median', 'kernel'):

      # this is a 3d array 
      groups = [[[] for i in range(7)] for j in range(24)]

      for i, row in enumerate(zip(hours, weekdays)):
        hour = int(row[0])
        day = int(row[1])
        groups[hour][day].append(y[i])

      if self.strategy == 'median':
        for i, j in np.ndindex((24, 7)):
          if groups[i][j]:
            a[i,j] = np.median(groups[i][j])
          else:
            a[i,j] = np.nan
      elif self.strategy == 'kernel':
        # kernel method computes a kernel density for each of the
        # bins and determines the most probably value ('mode' of sorts)
        grid = np.linspace(np.min(y), np.max(y), 1000)[:, np.newaxis]
        for i, j in np.ndindex((24, 7)):
          if groups[i][j]:
            npgroups = np.array(groups[i][j])[np.newaxis]
            kernel = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(npgroups.T)
            density = kernel.score_samples(grid)
            dmax = np.max(density)
            imax = np.where(density==dmax)
            a[i,j] = grid[imax, 0]
          else:
            a[i,j] = np.nan

      self._model = a

    # smooth the model here if there are nans

    #from matplotlib import pyplot as plt
    #plt.imshow(self._model)
    #plt.show()

    return self

  def predict(self, X):
    hours = np.copy(X[:, 1])
    weekdays = np.copy(X[:, 2])
    hours = 23 * normalize(hours)
    weekdays = 6 * normalize(weekdays)
    prediction = map(lambda x: self._model[x[0], x[1]], zip(hours, weekdays))
    return np.array(prediction)
