
import os
from skopt.space import Integer, Real, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize, dump
from sklearn.svm import SVC
from sklearn import clone
import numpy as np
import matplotlib.pyplot as plt

def load(working_dir = 'E:\\code\\VAE_keras',scale = 0.02, n_folds = 4):
  test = []
  train = []
  for f in range(n_folds):
    fold_path = os.path.join(working_dir,f'train_fold_{f}')
    # generated_path = os.path.join(fold_path,f'generated_latdim3_varScale_{scale}')
    train_path = os.path.join(fold_path,'train','validated')
    test_path = os.path.join(fold_path,'test','validated')

    types = os.listdir(test_path)
    types.remove('.DS_Store')

    test_data = []
    test_labels = []
    test_weights = []
    train_data = []
    train_labels = []
    train_weights = []

    for _type in types:
      test_type = os.path.join(test_path,_type)
      imgs = os.path.listdir(test_type)
      for img in imgs:
        test_data.append(plt.imread(img)[:,:,0])
        test_labels.append(_type)
        test_weights.append(1.0/len(imgs))

      train_type = os.path.join(train_path,_type)
      imgs = os.path.listdir(train_type)
      for img in imgs:
        train_data.append(plt.imread(img)[:,:,0])
        train_labels.append(_type)
        train_weights.append(1.0/len(imgs))

    test.append({'data':test_data,'labels':np.array(test_labels), 'weights':np.array(test_weights)})
    train.append({'data':train_data,'labels':np.array(train_labels), 'weights':np.array(train_weights)})    

  print('loaded all data')
  return test, train


def optimize(testing, training):
  n_folds = len(training)
  search_space = (
    Real(1e-6, 100.0, 'log-uniform', name='C'),
    Categorical(['linear','poly','rbf','sigmoid'], name='kernel'),
    Integer(1, 5, name='degree'),
    Real(1e-6, 100.0, 'log-uniform', name='gamma')
  )
  model = SVC()

  @use_named_args(search_space)
  def objective(**params):
    model.set_params(**params)
    accuracy = np.empty((n_folds,))
    for f in range(n_folds):
      clone(model).fit(training[f]['data'], training[f]['labels'], sample_weight=training[f]['weights'])
      accuracy[f] = np.mean((model.predict(testing[f]['data']) == testing[f]['labels']) @ testing['weights'])
    return 1.0 - np.mean(accuracy)

  def on_result(result):
    print('Iteration', len(result.x_iters))
    print('\tTested at', result.x_iters[-1])
    print('\tResult:', result.func_vals[-1])

  result = gp_minimize(objective, search_space, n_calls = 1000, callback=on_result)
  dump(result, 'results.pkl')

def main():
  test,train = load()
  optimize(test,train)

if __name__ == '__main__':
  main()