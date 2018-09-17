import numpy as np

with open('hero_winrate_python_obj.pkl', 'rb') as fp:
    hero_winrate = np.load(fp)

with open('hero_syn_vntg_python_obj.pkl', 'rb') as fp:
    hero_syn_vntg = np.load(fp)

with open('partidas_python_obj.pkl', 'rb') as fp:
    partidas = np.load(fp)


x = np.zeros((20,250))
y = np.zeros((20,1))


for i in range(20):

    count_index_x = 0

    y[i] = partidas[i][10]

    for hero in partidas[i][0:5]:
        x[i][count_index_x : count_index_x+16] = hero_winrate[hero]
        count_index_x += 16
        for hero_amigo in partidas[i][0:5]:
            if hero != hero_amigo:
                x[i][count_index_x] = hero_syn_vntg[hero][hero_amigo][0]
                count_index_x += 1
        for hero_inimigo in partidas[i][5:10]:
            x[i][count_index_x] = hero_syn_vntg[hero][hero_inimigo][1]
            count_index_x += 1

    for hero in partidas[i][5:10]:
        x[i][count_index_x : count_index_x+16] = hero_winrate[hero]
        count_index_x += 16
        for hero_amigo in partidas[i][5:10]:
            if hero != hero_amigo:
                x[i][count_index_x] = hero_syn_vntg[hero][hero_amigo][0]
                count_index_x += 1
        for hero_inimigo in partidas[i][0:5]:
            x[i][count_index_x] = hero_syn_vntg[hero][hero_inimigo][1]
            count_index_x += 1

print ("YEYEYEYEYEYAHHJHHH")