# -*-coding:utf-8-*-

import pandas as pd
import numpy as np


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer,TfidfTransformer
from collections import Counter

#read data
rec_data = pd.read_csv('recom_data_df.csv',encoding='gb18030')
rec_data.head(1)
len(rec_data)

#remove stopwords
stopwords = open('st.txt','r').readlines()[0].split(' ')


#prodinfo
infos = list(rec_data.product_info)
len(infos)
#remove stopwords
corpus = [' '.join([tm for tm in info.split() if tm not in stopwords]) for info in infos]
vectorizer = CountVectorizer(min_df=1000,stop_words='english')
# vectorizer = CountVectorizer(min_df=1000,stop_words=stopwords)
vect_count = vectorizer.fit_transform(corpus).toarray()
word = vectorizer.get_feature_names()
ct_df = pd.DataFrame(vect_count,columns=word)


ct_df.head(1)
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
# keywords in corpus
word = vectorizer.get_feature_names()
# tfidf
weight = tfidf.toarray()
weight.shape
len(corpus)


#rating according to user neighborhood
rec_data.head(1)
rate_df = rec_data[['review_user_name','product_name','review_score']]
user_df = pd.DataFrame(weight,columns=word)
data_df = pd.concat([rate_df,user_df],axis=1)
user_df['review_user_name'] = rate_df.review_user_name
ct_df['prod_name'] = rec_data.product_name
len(rate_df)
user_df.head(1)
ct_df = ct_df.drop_duplicates()
ct_df.head(1)
len(ct_df)

#tfidf
user_df.to_csv('user_df.csv',index=False)

#prod-feat_count
ct_df.to_csv('ct_df.csv',index=False)

#rec_data 
# rec_data.to_csv('recom_data_df.csv',index=False)


#recommendation
class recommender:
	# data：users
	# k：k nearest neighborhoods
	# metric：similarity 
	# n：n recommendations
	def __init__(self, data, user_infos, k=10, metric='cosine', n=5):

		self.k = k
		self.n = n
		self.username2id = {}
		self.userid2name = {}
		self.productid2name = {}
		self.userinfo = user_infos

		self.metric = metric
		if self.metric == 'cosine':
			self.fn = self.cosine
		if type(data).__name__ == 'dict':
			self.data = data

		# self.readData()

	def convertProductID2name(self, uid):
		if uid in self.productid2name:
			return self.productid2name[uid]
		else:
			return uid

	def cosine(self, rating1, rating2):
		sum_xy = 0
		sum_x = 0
		sum_y = 0
		sum_x2 = 0
		sum_y2 = 0
		n = 0

		for key in rating1:
			if key in rating2:
				n += 1
				x = rating1[key]
				y = rating2[key]
				sum_xy += x * y
				sum_x += x
				sum_y += y
				sum_x2 += pow(x, 2)
				sum_y2 += pow(y, 2)
		if n == 0:
			return 0

		dis = sum_xy/np.sqrt(sum_x2*sum_y2)
		if sum_x2*sum_y2 == 0:
			return 0
		else:
			return dis

	def computeNearestNeighbor(self, username,alpha=1):
		distances = []
		for instance in self.data:
			if instance != username:
				dist2 = np.sqrt(sum(np.square(np.array(self.userinfo[username]) - np.array(self.userinfo[instance]))))
				distance = alpha*self.fn(self.data[username], self.data[instance]) + (1-alpha)*dist2
				# print(username,instance,distance,'###############')
				distances.append((instance, distance))

		distances.sort(key=lambda artistTuple: artistTuple[1], reverse=True)
		return distances

	
	def recommend(self, user):
		# define documents
		recommendations = {}
		# calculate similarity between all users 
		nearest = self.computeNearestNeighbor(user)
		# print nearest
		userRatings = self.data[user]
		# print userRatings
		totalDistance = 0.0
		# distance between nearest neighborhoods
		for i in range(self.k):
			totalDistance += nearest[i][1]
		if totalDistance == 0.0:
			totalDistance = 1.0

		#items in neighborhoods
		recom_items = []
		# recommend items in the neighbrhood to user u
		for i in range(self.k):
			# similarity between user i and u
			weight = nearest[i][1] / totalDistance

			# i's name
			name = nearest[i][0]

			# user i rated items
			neighborRatings = self.data[name]
			# print(self.k,i,name,neighborRatings.keys())
			recom_items.extend(list(neighborRatings.keys()))


			# for artist in neighborRatings:
			# 	if not artist in userRatings:
			# 		if not artist in recommendations:
			# 			recommendations[artist] = neighborRatings[artist] * weight
			# 		else:
			# 			recommendations[artist] += neighborRatings[artist] * weight

		# recommendations = list(recommendations.items())
		# recommendations = {self.convertProductID2name(k): v for (k, v) in recommendations}

		
		# print(recommendations, '###############')
		# recom = dict(sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[0:self.n])

		# print(self.k,  '0')
		#items对应的feat
		items = set(recom_items)
		items_dict=dict()
		items_list = []
		for item in items:
			# print(item)
			# item ='Clinique Acne Solutions Clarifying Lotion'
			# ct_df.head(1)
			tmp1 = ct_df.loc[ct_df.prod_name==item,].as_matrix()[0][0:-1]
			# print(tmp1)
			# tmp = list(ct_df.iloc[1,:-1])
			tmp = [tx[0] for tx in list(zip(ct_df.columns,tmp1)) if tx[1]>0]
			items_dict[item] = tmp
			items_list.extend([tx[0] for tx in list(zip(ct_df.columns,tmp1)) if tx[1]>0])

		# print(self.k, '1')
		# items_dict={'a':['i2'],'b':['i1','i2'],'c':['i1','i2','i3']}
		# items_list =['i2','i2','i2','i1','i3','i1']
		count_dict = Counter(items_list)
		item_weight = dict()
		for k,v in items_dict.items():
			# print(k,v)
			item_weight[k] = sum([count_dict[tv] for tv in v])
		# print(self.k,item_weight)

		return item_weight
		# return recom, nearest

#recommend
def adjustrecommend(users,uname):
	r = recommender(users, user_infos)
	# recom_k, nearuser = r.recommend("%s" % id)
	recom_ = r.recommend(uname)
	return recom_


rate_df.head(1)
users = {}
for i in range(len(rate_df)):
	line = list(rate_df.ix[i,])
	if line[0] not in users:
		users[line[0]] = {}
	users[line[0]][line[1]] = float(line[2])


user_df.head(1)
user_infos = {}
for i in range(len(user_df)):
	line = list(user_df.ix[i,])
	if line[0] not in user_infos:
		user_infos[line[-1]] = []
		user_infos[line[-1]] = line[0:-1]

#
# 
# rd_users = []
# #random 50 users 
# total_rd_user_ct = 50
# while len(rd_users)<total_rd_user_ct:
# 	k = np.random.randint(1,len(user_infos.keys()))
# 	kname = list((user_infos.keys()))[k]
# 	if not kname in rd_users:
# 		rd_users.append(kname)
#
# #recommend
# recom_k = []
# # uname = 'lvoegeli'
# for uname in rd_users:
# 	recom_ = adjustrecommend(users,uname)
# 	sorted_recom_ = dict(sorted(recom_.items(), key=lambda x: x[1], reverse=True)[0:3])
# 	for k, v in sorted_recom_.items():
# 		print(k, '\t', v)
# 		recom_k.append((uname,k,v))
#
# #output
# res = pd.DataFrame(recom_k,columns=['uname','prod','score'])
# res.head(2)
# len(res)
# res.to_csv('E:/work_dir/tasks/task39_Aa/res.csv',index=False)


#train test split
n_u = 100
n_p = 5
len(users.keys())
len(user_infos.keys())

M = len(user_df)
N = len(ct_df)
L = 5


new_rd_users = []
#random 100 users who rated more than 5 items
total_rd_user_ct = 50

info_key = user_infos.keys()
while len(new_rd_users)<total_rd_user_ct:
	k = np.random.randint(1,len(info_key))
	# print('随机数：',k)
	kname = list(info_key)[k]
	if not kname in new_rd_users:
		B = len(users[kname])
		if B >=5:
			new_rd_users.append(kname)

print('100个用户:',new_rd_users)
#2:8
all5_accs = []
all5_recalls = []
for ct in range(5):
	all_accs = []
	all_recalls = []

	# train-test-split
	from sklearn.model_selection import train_test_split
	new_train_rate_df = pd.DataFrame(columns=rate_df.columns)
	new_test_rate_df = pd.DataFrame(columns=rate_df.columns)
	for tname in new_rd_users:
		tdf = rate_df.loc[rate_df['review_user_name'] == tname,]
		trainf,testf = train_test_split(tdf,test_size=0.2)
		new_train_rate_df = pd.concat([new_train_rate_df,trainf],axis=0)
		new_test_rate_df = pd.concat([new_test_rate_df,testf],axis=0)
	#reset index
	new_train_rate_df = new_train_rate_df.reset_index(drop=True)
	new_test_rate_df = new_test_rate_df.reset_index(drop=True)
	new_rate_df = pd.concat([new_train_rate_df,new_test_rate_df],axis=0)

	
	
	train_users = {}
	for i in range(len(new_train_rate_df)):
		line = list(new_train_rate_df.ix[i,])
		if line[0] not in train_users:
			train_users[line[0]] = {}
		train_users[line[0]][line[1]] = float(line[2])

	
	# user_df.head(1)
	# user_infos = {}
	# for i in range(len(user_df)):
	# 	line = list(user_df.ix[i,])
	# 	if line[0] not in user_infos:
	# 		user_infos[line[-1]] = []
	# 		user_infos[line[-1]] = line[0:-1]


	train_recom_k = []
	# uname = 'lvoegeli'
	for uname in train_users:
		# print(uname)
		recom_ = adjustrecommend(train_users,uname)
		sorted_recom_ = dict(sorted(recom_.items(), key=lambda x: x[1], reverse=True)[0:10])
		
		sorted_recom_items = set(sorted_recom_.keys())

		new_test_rate_df.head(1)
		test_items = set(new_test_rate_df.loc[new_test_rate_df.review_user_name==uname,]['product_name'])
		# test_items = set(new_rate_df.loc[new_rate_df.review_user_name==uname,]['product_name'])
		
		tp = len(sorted_recom_items.intersection(test_items))
		#precesion 
		acc = tp/len(sorted_recom_items)
		#recall
		recall = tp/len(test_items)
		# for k, v in sorted_recom_.items():
		# 	print(k, '\t', v)
		# 	train_recom_k.append((uname,k,v))
		all_accs.append(acc)
		all_recalls.append(recall)
	all5_accs.append(np.mean(all_accs))
	all5_recalls.append(np.mean(all_recalls))
	print('acc:{a},recall:{b}'.format(a=np.mean(all_accs),b=np.mean(all_recalls)))

print('all5_accs:{a},all5_recalls:{b}'.format(a=np.mean(all5_accs),b=np.mean(all5_recalls)))

print('100 users rated items：',len(new_rate_df))




recom_k = []
for uname in new_rd_users:
	recom_ = adjustrecommend(users,uname)
	sorted_recom_ = dict(sorted(recom_.items(), key=lambda x: x[1], reverse=True)[0:3])
	for k, v in sorted_recom_.items():
		print(k, '\t', v)
		recom_k.append((uname,k,v))
pd.DataFrame(recom_k).to_csv('recom50.csv')
