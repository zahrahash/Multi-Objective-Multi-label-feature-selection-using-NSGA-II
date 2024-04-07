from skmultilearn.dataset import load_dataset
from skmultilearn.dataset import available_data_sets
from skmultilearn.model_selection import iterative_train_test_split


def read_datas_from_Mulan(d_name): 

    mulan_datasets = [x[0] for x in available_data_sets().keys()]
    if d_name not in mulan_datasets:
        raise ValueError('{} not found in Mulan database'.format(d_name)) 
    X, y, _ , _  = load_dataset(d_name, 'undivided')
    t_size = 0.2
    X_train, Y_train, X_test, Y_test = iterative_train_test_split(X, y, t_size)
    return X_train.toarray(), Y_train.toarray(), X_test.toarray(), Y_test.toarray()