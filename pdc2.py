import h2o
from h2o.estimators import H2ODeepLearningEstimator


h2o.init()
df = h2o.import_file(path='house-votes-84.csv')
train, valid = df.split_frame(ratios=[.7])

for i, neuron_type in enumerate(['Tanh', 'Rectifier']):
    for j, k in enumerate([0, 5, 10]):
        name = 'votes_neurons={}_folds_in_cross_validation={}'.format(neuron_type, k)
        hh = H2ODeepLearningEstimator(model_id=name,
                                      activation=neuron_type,
                                      loss='CrossEntropy',
                                      hidden=[5, 20, 100],
                                      epochs=6666,
                                      nfolds=k)

        hh.train(x=list(range(16)), y=16, training_frame=train, validation_frame=valid)
        hh.show()
