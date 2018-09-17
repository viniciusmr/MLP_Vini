import pickle
import mysql.connector as mariadb
import numpy as np


def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


mariadb_connection = mariadb.connect(user='root', password='uc775ft', database='tcc')
cursor = mariadb_connection.cursor()
## cursor.execute("SELECT * FROM hero_names WHERE hero_id > 30 LIMIT %s", (10,))

# ==================================================================================================================================

# cursor.execute("SELECT * FROM hero_winrate")
# hero_winrate = []
# hero_winrate.append(None)

# for hero_id, avg_wr, wr15, wr19, wr23, wr27, wr31, wr35, wr39, wr43, wr47, wr51, wr55, wr59, wr63, wr67, wr71 in cursor:
#     hero_winrate.append([avg_wr, wr15, wr19, wr23, wr27, wr31, wr35, wr39, wr43, wr47, wr51, wr55, wr59, wr63, wr67, wr71])

# # print (hero_winrate[20])
## save_object(hero_winrate, 'hero_winrate_python_obj.pkl')
# ==================================================================================================================================

cursor.execute("SELECT * FROM hero_v_hero_normalized")
hero_syn_vntg = np.zeros((113,113,2))
for hero_id, target_hero_id, sinergia, vantagem in cursor:
    if sinergia is not None:
        hero_syn_vntg[hero_id][target_hero_id] = [sinergia, vantagem]
        hero_syn_vntg[target_hero_id][hero_id] = [sinergia, vantagem*(-1)]

save_object(hero_syn_vntg, 'hero_syn_vntg_python_obj.pkl')
print (hero_syn_vntg[2][27])
print (hero_syn_vntg[27][2])
# ==================================================================================================================================

# cursor.execute("SELECT hero0, hero1, hero2, hero3, hero4, hero5, hero6, hero7, hero8, hero9, radiant_win FROM matches_high_skill")

# partidas = []

# for hero0, hero1, hero2, hero3, hero4, hero5, hero6, hero7, hero8, hero9, radiant_win in cursor:
#     if radiant_win == 'f':
#         radiant_win = 0
#     else:
#         radiant_win = 1
#     partidas.append([hero0, hero1, hero2, hero3, hero4, hero5, hero6, hero7, hero8, hero9, radiant_win])

# print (len(partidas))
# save_object(partidas, 'partidas_python_obj.pkl')
# ==================================================================================================================================

# with open(r"someobject.pickle", "rb") as input_file:
#    ...:     e = cPickle.load(input_file)


# import pickle
# with open('Fruits.obj', 'rb') as fp:
#     banana = pickle.load(fp)