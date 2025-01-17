
import torch
import numpy as np
import argparse
import time
import util
import os.path
#import matplotlib.pyplot as plt
from engine import trainer
import generate_training_data_drb

parser = argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='cuda',help='')
parser.add_argument('--data',type=str,default='data/DRB_gwn_full',help='data path')
parser.add_argument('--adjdata',type=str,default='data/DRB_gwn_full/adj_mx.pkl',help='adj data path')
parser.add_argument('--adjtype',type=str,default='doubletransition',help='adj type')
parser.add_argument('--gcn_bool',action='store_true',help='whether to add graph convolution layer')
parser.add_argument('--aptonly',action='store_true',help='whether only adaptive adj')
parser.add_argument('--addaptadj',action='store_true',help='whether add adaptive adj')
parser.add_argument('--randomadj',action='store_true',help='whether random initialize adaptive adj')
parser.add_argument('--seq_length',type=int,default=365,help='')
parser.add_argument('--nhid',type=int,default=32,help='')
parser.add_argument('--in_dim',type=int,default=8,help='inputs dimension')
parser.add_argument('--out_dim',type=int, default=12,help = 'Period of output predictions')
parser.add_argument('--num_nodes',type=int,default=456,help='number of nodes')
parser.add_argument('--batch_size',type=int,default=1,help='batch size')
parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
parser.add_argument('--epochs',type=int,default=50,help='')
parser.add_argument('--epochs_pre',type=int,default=25,help='')
parser.add_argument('--print_every',type=int,default=20,help='')
'''
#parser.add_argument('--seed',type=int,default=99,help='random seed')
parser.add_argument('--save',type=str,default='./train_val_drb/',help='save path')
parser.add_argument('--expid',type=str,default='default',help='experiment id')
parser.add_argument('--kernel_size',type=int, default=2)
parser.add_argument('--layer_size',type=int,default=2)
parser.add_argument('--n_blocks',type=int, default=4)
## data args
parser.add_argument("--obs_temper_file", type=str, default='../river-dl/data/in/obs_temp_full')
parser.add_argument('--obs_flow_file',type=str, default='../river-dl/data/in/obs_flow_full')
parser.add_argument("--pretrain_file",type=str,default='../river-dl/data/in/uncal_sntemp_input_output')
parser.add_argument('--train_start_date',nargs='+', default= ['1985-10-01', '2016-10-01'])
parser.add_argument("--train_end_date",nargs='+',default=['2006-09-30', '2020-09-30'])
parser.add_argument("--val_start_date",type=str, default='2006-10-01')
parser.add_argument("--val_end_date",type=str,default='2016-09-30')
parser.add_argument("--test_start_date",nargs='+',default=['1980-10-01', '2020-10-01'])
parser.add_argument("--test_end_date",nargs='+',default=['1985-09-30', '2021-09-30'])
parser.add_argument("--x_vars",nargs='+',default=["seg_rain", "seg_tave_air", "seginc_swrad", "seg_length", "seginc_potet", "seg_slope", "seg_humid","seg_elev"])
parser.add_argument("--y_vars",nargs="+",default=['seg_tave_water'])
parser.add_argument("--primary_variable",type=str,default='temp')
parser.add_argument("--seq_length",type=int,default=365)
parser.add_argument("--period=np.nan,
    offset=1,
    out_file = 'data/DRB_gwn'

'''
args = parser.parse_args()
'''
args = parser.parse_args(['--epochs', '1'])
args.data = 'data/DRB_gwn'
args.adjdata = 'data/DRB_gwn/adj_mx.pkl'
args.adjtype = 'transition'
args.device = 'cpu'
args.out_dim=365
args.seq_length=365
#args.gcn_bool = True
#args.addaptadj = True
args.num_nodes = 42
args.epochs_pre = 1
args.batch_size = 2
args.expid='testsub'
args.kernel_size = 4
args.layer_size = 6
'''

def main():
    #set seed
    #torch.manual_seed(args.seed)
    #np.random.seed(args.seed)
    #load data
    device = torch.device(args.device)
    sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(args.adjdata,args.adjtype)
    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']
    supports = [torch.tensor(i).to(device) for i in adj_mx]

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    print(args)

    if args.randomadj:
        adjinit = None
    else:
        adjinit = supports[0]

    if args.aptonly:
        supports = None



    engine = trainer(scaler, args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                         args.learning_rate, args.weight_decay, device, supports, args.gcn_bool, args.addaptadj,
                         adjinit, args.out_dim, args.kernel_size, args.n_blocks, args.layer_size)

    print("start pre_training...", flush=True)
    phis_loss =[]
    pval_time = []
    ptrain_time = []
    for i in range(1,args.epochs_pre+1):
        #if i % 10 == 0:
            #lr = max(0.000002,args.learning_rate * (0.1 ** (i // 10)))
            #for g in engine.optimizer.param_groups:
                #g['lr'] = lr
        train_loss = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        dataloader['pre_train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader['pre_train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device)
            trainx= trainx.transpose(1, 3)
            trainy = torch.Tensor(y).to(device)
            trainy = trainy.transpose(1, 3)
            metrics = engine.train(trainx, trainy[:,0,:,:])
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            if iter % args.print_every == 0 :
                log = 'Pre_Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]),flush=True)
        t2 = time.time()
        ptrain_time.append(t2-t1)
        #torch.save(engine.model.state_dict(), args.save+args.expid+"pre_epoch_"+str(i)+"_.pth")
    print("Average Pre_Training Time: {:.4f} secs/epoch".format(np.mean(ptrain_time)))
    #print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    print("start training...",flush=True)
    his_loss =[]
    val_time = []
    train_time = []
    for i in range(1,args.epochs+1):
        #if i % 10 == 0:
            #lr = max(0.000002,args.learning_rate * (0.1 ** (i // 10)))
            #for g in engine.optimizer.param_groups:
                #g['lr'] = lr
        train_loss = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device)
            trainx= trainx.transpose(1, 3)
            trainy = torch.Tensor(y).to(device)
            trainy = trainy.transpose(1, 3)
            metrics = engine.train(trainx, trainy[:,0,:,:])
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            if iter % args.print_every == 0 :
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]),flush=True)
        t2 = time.time()
        train_time.append(t2-t1)
        #validation
        valid_loss = []
        valid_mape = []
        valid_rmse = []


        s1 = time.time()
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(device)
            testy = testy.transpose(1, 3)
            metrics = engine.eval(testx, testy[:,0,:,:])
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i,(s2-s1)))
        val_time.append(s2-s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        his_loss.append(mvalid_loss)

        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)),flush=True)
        torch.save(engine.model.state_dict(), args.save+'/tmp/'+args.expid+"_epoch_"+str(i)+"_"+str(round(mvalid_loss,2))+".pth")
    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    #testing
    bestid = np.argmin(his_loss)
    engine.model.load_state_dict(torch.load(args.save+'/tmp/'+args.expid+"_epoch_"+str(bestid+1)+"_"+str(round(his_loss[bestid],2))+".pth"))


    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1,3)[:,0,:,:]

    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1,3)
        with torch.no_grad():
            preds = engine.model(testx).transpose(1,3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs,dim=0)
    yhat = yhat[:realy.size(0),...]


    print("Training finished")
    print("The valid loss on best model is", str(round(his_loss[bestid],4)))


    amae = []
    amape = []
    armse = []
    for i in range(12):
        #pred = scaler.inverse_transform(yhat[:,:,i])
        pred = yhat[:,:,i]
        real = realy[:,:,i]
        metrics = util.metric(pred,real)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i+1, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])

    log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(np.mean(amae),np.mean(amape),np.mean(armse)))
    torch.save(engine.model.state_dict(), args.save+args.expid+"_best_"+str(round(his_loss[bestid],2))+".pth")



if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2-t1))
