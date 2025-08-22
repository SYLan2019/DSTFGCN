import numpy as np
import os
import pickle
import csv

def get_adjacency_matrix(distance_df_filename, num_of_vertices,
                         type_='connectivity', id_filename=None):
    A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                 dtype=np.float32)
    if id_filename:
        with open(id_filename, 'r') as f:
            id_dict = {int(i): idx
                       for idx, i in enumerate(f.read().strip().split('\n'))}
        with open(distance_df_filename, 'r') as f:
            f.readline()
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 3:
                    continue
                i, j, distance = int(row[0]), int(row[1]), float(row[2])
                A[id_dict[i], id_dict[j]] = 1
                A[id_dict[j], id_dict[i]] = 1
        return A

    with open(distance_df_filename, 'r') as f:
        f.readline()
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 3:
                continue
            i, j, distance = int(row[0]), int(row[1]), float(row[2])
            if type_ == 'connectivity':
                A[i, j] = 1
                A[j, i] = 1
            elif type == 'distance':
                A[i, j] = 1 / distance
                A[j, i] = 1 / distance
            else:
                raise ValueError("type_ error, must be "
                                 "connectivity or distance!")
    return A

def generate_seq(data, train_length, pred_length):
    seq = np.concatenate([np.expand_dims(
        data[i: i + train_length + pred_length], 0)
        for i in range(data.shape[0] - train_length - pred_length + 1)],
        axis=0)[:, :, :, 0: 1]
    return np.split(seq, 2, axis=1)

def generate_from_train_val_test(data, transformer):
    mean = None
    std = None
    for key in ('train', 'val', 'test'):
        x, y = generate_seq(data[key], 12, 12)
        if transformer:
            x = transformer(x)
            y = transformer(y)
        if mean is None:
            mean = x.mean()
        if std is None:
            std = x.std()
        yield (x - mean) / std, y

def generate_from_data(data, length, transformer):
    mean = None
    std = None
    train_line, val_line = int(length * 0.6), int(length * 0.8)
    for line1, line2 in ((0, train_line),
                         (train_line, val_line),
                         (val_line, length)):
        x, y = generate_seq(data['data'][line1: line2], 12, 12)
        if transformer:
            x = transformer(x)
            y = transformer(y)
        if mean is None:
            mean = x.mean()
        if std is None:
            std = x.std()
        yield (x - mean) / std, y

def generate_data(graph_signal_matrix_filename, transformer=None):
    data = np.load(graph_signal_matrix_filename)
    print("data type",type(data))
    print("data len",len(data))
    keys = data.keys()
    print("data keys",keys)
    if 'train' in keys and 'val' in keys and 'test' in keys:
        for i in generate_from_train_val_test(data, transformer):
            yield i
    elif 'data' in keys:
        length = data['data'].shape[0]
        for i in generate_from_data(data, length, transformer):
            yield i
    else:
        raise KeyError("neither data nor train, val, test is in the data")


seq_length_x =12
seq_length_y =12

if __name__ == '__main__':
    adj_matrix = []
    num_nodes = 0
    x_list = []
    y_list = []

    for idx, (x, y) in enumerate(generate_data("./data/PEMS07/PEMS07.npz")):
        print(x.shape, y.shape)
        x_list.append(x)
        y_list.append(y)

    x_train = x_list[0]
    y_train = y_list[0]
    x_val = x_list[1]
    y_val = y_list[1]
    x_test = x_list[2]
    y_test = y_list[2]

    x_offsets = np.sort(np.concatenate((np.arange(-(seq_length_x - 1), 1, 1),)))
    y_offsets = np.sort(np.arange(1, (seq_length_y + 1), 1))
    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join("data/PEMS07", f"{cat}.npz"),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )

    num_nodes = x_train.shape[2]
    print("num_nodes",num_nodes)
    adj_list = []
    adj_list.append([])
    adj_list.append([])
    adj = get_adjacency_matrix(distance_df_filename="./data/PEMS07/PEMS07.csv",
                               num_of_vertices=num_nodes,id_filename=None)
    '''
    If processing datasets other than PEMS03, write 'None' in the 'id_filename=' field
    '''
    adj_list.append(adj)
    with open("data/graph/adj_O7.pkl", "wb") as f:
        pickle.dump(adj_list, f)



