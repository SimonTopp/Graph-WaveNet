import util
import argparse
from model import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


args = argparse.Namespace(addaptadj=True, adjdata='data/DRB_gwn_full/adj_mx.pkl', adjtype='transition', aptonly=False, batch_size=8, data='data/DRB_gwn_full_60', device='cuda', dropout=0.3, epochs=100, epochs_pre=50, expid='full60', gcn_bool=True, in_dim=8, kernel_size=4, layer_size=3, learning_rate=0.001, n_blocks=4, nhid=32, num_nodes=456, out_dim=30, print_every=10, randomadj=True, save='./train_val_drb/', seq_length=60, weight_decay=0.0001)
args.checkpoint = './train_val_drb/full60_best_1.56.pth'
args.device = 'cpu'


_, _, adj_mx = util.load_adj(args.adjdata,args.adjtype)
supports = [torch.tensor(i).to(args.device) for i in adj_mx]
if args.randomadj:
    adjinit = None
else:
    adjinit = supports[0]

if args.aptonly:
    supports = None
model = gwnet(args.device, args.num_nodes, args.dropout, supports=supports, gcn_bool=args.gcn_bool, addaptadj=args.addaptadj,
               in_dim=args.in_dim, out_dim=args.out_dim, residual_channels=args.nhid, dilation_channels=args.nhid,
              skip_channels=args.nhid * 8, end_channels=args.nhid * 16, aptinit=adjinit, kernel_size=args.kernel_size, layers=args.layer_size)

#model =  gwnet(device, args.num_nodes, args.dropout, supports=supports, gcn_bool=args.gcn_bool, addaptadj=args.addaptadj, aptinit=adjinit)
model.to(args.device)
#dict = torch.load(args.checkpoint, map_location=torch.device('cpu'))
model.load_state_dict(torch.load(args.checkpoint, map_location=torch.device('cpu')))
model.eval()

print('model load successfully')

dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
scaler = dataloader['scaler']
outputs = []
realy = torch.Tensor(dataloader['y_test']).to(args.device)
realy = realy.transpose(1,3)[:,0,:,:]

for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
    testx = torch.Tensor(x).to(args.device)
    testx = testx.transpose(1,3)
    with torch.no_grad():
        preds = model(testx).transpose(1,3)
    outputs.append(preds.squeeze())

yhat = torch.cat(outputs,dim=0)
yhat = yhat[:realy.size(0),...]
print(yhat.shape)

'''
adp = F.softmax(F.relu(torch.mm(model.nodevec1, model.nodevec2)), dim=1)
device = torch.device('cpu')
adp.to(device)
adp = adp.cpu().detach().numpy()
adp = adp*(1/np.max(adp))
df = pd.DataFrame(adp)
plt.imshow(df, cmap="YlBu")
plt.show()
'''

adp = F.softmax(F.relu(torch.mm(model.nodevec1, model.nodevec2)), dim=1)
device = torch.device('cpu')
adp.to(device)
adp = adp.cpu().detach().numpy()
adp = adp*(1/np.max(adp))
df = pd.DataFrame(adp)
sns.heatmap(df, cmap="RdYlBu")
        #plt.savefig("./emb"+ '.pdf')


data = np.load(args.data + '/data.npz')
period = data['period'][0]

test_dates =  np.transpose(data['dates_test'],(0,3,2,1)).squeeze()
test_ids =  np.transpose(data['ids_test'],(0,3,2,1)).squeeze()

if ~np.isnan(period):
    test_ids = test_ids[:,:,-period:]
    test_dates = test_dates[:,:,-period:]


def prepped_array_to_df(data_array, obs, dates, ids):

    df_obs = pd.DataFrame(obs.flatten(), columns = ['temp_ob'])
    df_preds = pd.DataFrame(data_array.flatten(), columns=['temp_pred'])
    df_dates = pd.DataFrame(dates.flatten(), columns=["date"])
    df_ids = pd.DataFrame(ids.flatten(), columns=["seg_id_nat"])
    df = pd.concat([df_dates, df_ids, df_preds, df_obs], axis=1)
    return df

test_df = prepped_array_to_df(np.array(yhat), np.array(realy), test_dates, test_ids)

counts = test_df.dropna().groupby('seg_id_nat').size()
filt = counts[counts > 1000].index.tolist()
test_df = test_df[test_df.seg_id_nat.isin(filt)]


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def plotter(df,seg):
    df = df[df.seg_id_nat == seg]
    actual = df.temp_ob
    predicted = df.temp_pred
    fig, ax = plt.subplots()
    x = df.date
    ax.scatter(x, actual, s=2, alpha = .5, label='Actual')
    ax.scatter(x, predicted,s=2, alpha=.5, label='Predicted')
    rms = rmse(actual, predicted)
    ax.legend()
    ax.set_title(str(seg) + ': RMSE ' + str(rms))


plotter(test_df,1498)
names = ['x', 'y', 'z']
index = pd.MultiIndex.from_product([range(s)for s in realy.shape], names=names)
df = pd.DataFrame({'realy': realy.flatten()}, index=index)['realy']
df = df.unstack(level='y').swaplevel().sort_index()
df.index.names = ['DOY', 'Year']
df.reset_index(inplace=True)

_, ids,_ = util.load_adj(args.adjdata,args.adjtype)
segIds = list(ids.keys())
df.columns = ['DOY','Year'] +segIds
df = df.dropna(axis=1,thresh=200)
df = df.sort_values(by=['Year','DOY'])

predicted = scaler.inverse_transform(yhat)
index = pd.MultiIndex.from_product([range(s)for s in predicted.shape], names = names)
dfpreds = pd.DataFrame({'predicted':predicted.flatten()}, index = index)['predicted']
dfpreds = dfpreds.unstack(level = 'y').swaplevel().sort_index()
dfpreds.index.names = ["DOY", 'Year']
dfpreds.reset_index(inplace=True)
dfpreds.columns = ["DOY","Year"] +segIds
dfpreds = dfpreds[list(df.columns)]
dfpreds = dfpreds.sort_values(by=['Year','DOY'])

def plotter(seg):
    actual = df[seg]
    predicted = dfpreds[seg]
    fig, ax = plt.subplots()
    x = range(len(actual))
    ax.plot(x,actual, label = 'Actual')
    ax.plot(x,predicted, label = 'Predicted')
    ax.legend()

df.columns
plotter('1703')
dfpreds.filter('Year' is 1)