import torch
import numpy as np
import argparse
import time
import util
from engine import trainer
from pathlib import Path
import logging
import os
import pickle
import json

def save_model(path: str, **save_dict):
    os.makedirs(os.path.split(path)[0], exist_ok=True)
    torch.save(save_dict, path)

def load_dataset_config(config_path):
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='config/PeMS08.json', help='config_file')
    parser.add_argument('--device', type=str, default='cuda:0', help='gpu')
    args = parser.parse_args()
    config_path = args.config_path
    config = load_dataset_config(config_path)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    Path("log/").mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(f"log/{str(time.time())}_restore_{'STDGCN'}.log")
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(config)

    if not os.path.exists(config['save']):
        os.makedirs(config['save'])
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    device = torch.device(args.device)
    dataloader = util.load_dataset(config['data'], config['N_t'], config['batch_size'], config['batch_size'], config['batch_size'])
    with open(config['adjacency matrix'], 'rb') as file:
        adjacency_matrix = pickle.load(file)
    scaler = dataloader['scaler']
    engine = trainer(config, scaler, adjacency_matrix[2], args.device)

    print("start training...", flush=True)
    his_loss = []
    val_time = []
    train_time = []
    count = 0
    mtrain_loss_list = []
    mval_loss_list = []

    for i in range(1, config['max_epoch'] + 1):
        train_loss = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        for iter, (x, y, d) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device)
            trainx = trainx.transpose(1, 3)
            trainy = torch.Tensor(y).to(device)
            trainy = trainy.transpose(1, 3)
            metrics = engine.train(trainx, trainy[:, 0, :, :], d)
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            if iter % config['print_every'] == 0:
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                logger.info(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]))
        engine.scheduler.step()
        t2 = time.time()
        train_time.append(t2 - t1)
        valid_loss = []
        valid_mape = []
        valid_rmse = []

        s1 = time.time()
        for iter, (x, y, d) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(device)
            testy = testy.transpose(1, 3)
            metrics = engine.eval(testx, testy[:, 0, :, :], d)
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        logger.info(log.format(i, (s2 - s1)))
        val_time.append(s2 - s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)
        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        mtrain_loss_list.append(mtrain_loss)
        mval_loss_list.append(mvalid_loss)

        if len(his_loss) > 0 and mvalid_loss < np.min(his_loss):
            count = 0
        else:
            count += 1
        his_loss.append(mvalid_loss)
        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        logger.info(
            log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)))
        torch.save(engine.model.state_dict() , config['save'] + 'DSTFGCN' + "_epoch_" + str(i) + "_" + str(round(mvalid_loss, 2)) + ".pth")
        if count >= config['patience']:
            break
    logger.info("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    logger.info("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    bestid = np.argmin(his_loss)
    engine.model.load_state_dict(torch.load(config['save'] + 'DSTFGCN' + "_epoch_" + str(bestid + 1) + "_" + str(round(his_loss[bestid], 2)) + ".pth"))
    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1, 3)[:, 0, :, :]

    for iter, (x, y, d) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)
        with torch.no_grad():
            preds = engine.model(testx, d).transpose(1, 3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]
    logger.info("Training finished")
    log = 'The valid loss on best model is {:.4f}'
    logger.info(log.format(his_loss[bestid]))

    amae = []
    amape = []
    armse = []
    for i in range(12):
        pred = scaler.inverse_transform(yhat[:, :, i])
        real = realy[:, :, i]
        metrics = util.metric(pred, real)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        logger.info(log.format(i + 1, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])

    log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    logger.info(log.format(np.mean(amae), np.mean(amape), np.mean(armse)))
    torch.save(engine.model.state_dict(), config['save'] + 'DSTFGCN' + "_exp" + str(config['expid']) + "_best_" + str(round(his_loss[bestid], 2)) + ".pth")
    logger.info(config)

if __name__ == "__main__":
    main()

