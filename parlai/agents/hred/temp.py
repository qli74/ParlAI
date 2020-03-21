import pickle
with open('../../../data/light_dialogue/light_data.pkl','rb') as fp:
    light=pickle.load(fp)

print(light[0])

