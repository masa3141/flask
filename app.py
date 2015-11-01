from flask import Flask, jsonify
from mf import MF
import numpy as np
import threading
import pandas as pd
app = Flask(__name__)


class MfApp():
    def __init__(self):

        # load data
        print 'start load data'
        data = pd.read_csv('data/movielens/user_ratedmovies.dat', delimiter='\t')
        item_info = pd.read_csv('data/movielens/movies.dat', delimiter='\t')
        self.itemid2name = dict(zip(item_info['id'].tolist(), item_info['title'].tolist()))

        N = len(set(data['userID'].tolist()))  # number of user
        M = len(set(data['movieID'].tolist()))  # number of movie

        rating_matrix = np.zeros([N, M])
        userid2index = {}
        itemid2index = {}
        userid2itemindexes = {}

        for i, row in data.iterrows():
            userid = row['userID']
            itemid = row['movieID']
            rating = row['rating']
            # print userid, itemid, rating
            if userid in userid2index:
                userindex = userid2index[userid]
                userid2itemindexes[userid].append(itemid)
            else:
                userindex = len(userid2index)
                userid2index[userid] = userindex
                userid2itemindexes[userid] = [itemid]

            if itemid in itemid2index:
                itemindex = itemid2index[itemid]
            else:
                itemindex = len(itemid2index)
                itemid2index[itemid] = itemindex

            rating_matrix[userindex, itemindex] = rating

        self.userid2itemindexes = userid2itemindexes
        self.userid2index = userid2index
        self.itemid2index = itemid2index
        self.index2userid = {y: x for x, y in userid2index.items()}
        self.index2itemid = {y: x for x, y in itemid2index.items()}

        nonzero_col, nonzero_row = rating_matrix.nonzero()
        inds = zip(nonzero_col.tolist(), nonzero_row.tolist())
        print 'finish load data'
        K = 10
        alpha = 0.0001
        lam = 0.01
        self.mf = MF(rating_matrix, inds, K, alpha, lam)
        self.is_training = False
        self.losses = []
        self.epochs = []


class StartThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        print "start thread"
        mfapp.is_training = True
        mfapp.epochs = []
        for i in range(10):
            mfapp.mf.train(epochs=1)
            mfapp.epochs.append(i)
            mfapp.losses.append(mfapp.mf.rmse())
        mfapp.is_training = False
        print "end thread"


mfapp = MfApp()

@app.route('/')
def hello_world():
    li = [
        {'param': 'foo', 'val': 2},
        {'param': 'bar', 'val': 10}
        ]
    return jsonify({'results': li})


@app.route('/start/')
def start_training():
    response = {}
    response['start'] = 'NO'
    response['message'] = 'Now training.'
    if mfapp.is_training is False:
        response['start'] = 'YES'
        response['message'] = 'Start trainig'
        th = StartThread()
        th.start()
    return jsonify(response)


@app.route('/status/')
def status_trainig():
    response = {}
    response['satus'] = 'finished'
    if mfapp.is_training is True:
        response['satus'] = 'training'

    response['epochs'] = []
    for i in range(len(mfapp.epochs)):
        info = {}
        info['epoch'] = mfapp.epochs[i]
        info['loss'] = mfapp.losses[i]
        response['epochs'].append(info)
    response['process'] = len(mfapp.epochs)*1.0 / 10
    return jsonify(response)


@app.route('/users/')
def show_users():
    print "users"
    return "User"


@app.route('/users/<int:id>')
def show_user(id):
    response = {}
    response['id'] = id
    userindex = mfapp.userid2index[id]
    response['index'] = userindex
    R_predict = mfapp.mf.predict()
    user_predict = R_predict[userindex, :]
    top_item_indexes = np.argsort(user_predict)[::-1][:10]
    response['rated_items'] = []
    for itemid in mfapp.userid2itemindexes[id]:
        info = {}
        info['id'] = itemid
        info['name'] = unicode(mfapp.itemid2name[itemid], errors='replace')
        response['rated_items'].append(info)
    response['recommended_items'] = []
    for i, itemindex in enumerate(top_item_indexes):
        info = {}
        itemid = mfapp.index2itemid[itemindex]
        info['id'] = itemid
        info['index'] = itemindex
        info['name'] = unicode(mfapp.itemid2name[itemid], errors='replace')
        info['rank'] = i+1
        info['score'] = user_predict[itemindex]
        response['recommended_items'].append(info)
    print response
    print response['rated_items'][0]
    print response['recommended_items'][0]
    return jsonify(response)


@app.route('/items/')
def show_items():
    print "items"
    return jsonify(results=[1, 2])


@app.route('/items/<int:id>')
def show_item(id):
    response = {}
    response['id'] = id
    itemindex = mfapp.itemid2index[id]
    response['index'] = itemindex
    response['name'] = mfapp.itemid2name[id]
    response['latent_factor'] = mfapp.mf.q[itemindex, :].tolist()
    return jsonify(response)




if __name__ == '__main__':
    app.run(debug=True)
