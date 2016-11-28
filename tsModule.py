import csv
import numpy as np

class OzoneData:
   def __init__(self):
      self.x = []
      self.y = []
      with open('Ozone.csv', 'r') as f:
         reader = csv.DictReader(f)
         for row in reader:
            self.x.append(row['Month'])
            self.y.append(row['Ozone_Thickness'])
         self.n_obs = len(self.x)
         self.y = np.array(list(map(float, self.y)))/100.0

   # returns a [n_periods] array of the time series starting at pos
   def get_input(self, pos, n_periods):
      if pos + n_periods >= self.n_obs:
         print('WARNING')
      return self.y[pos:pos+n_periods]

   # returns a [n_batches, n_periods] array of random batches of specified n_periods
   def random_batch(self, n_batches, n_periods):

      # get random positions
      rand_pos = np.random.randint(0, self.n_obs-n_periods, (n_batches))
      stack = self.get_input(rand_pos[0], n_periods)
      target = self.y[rand_pos[0]+n_periods]

      for i in range(1, n_batches):
         stack = np.vstack( (stack, self.get_input(rand_pos[i], n_periods)))
         target = np.vstack( (target, self.y[rand_pos[i]+n_periods]) )

      return stack, target