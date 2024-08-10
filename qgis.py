from PIL import Image
import nu
img_plt_1 = np.array(Image.open('bolivia_23014_labelhand.png').convert('RGB'))
img_plt_2 = np.array(Image.open('bolivia_76104_labelhand.png').convert('RGB'))
Y_list = []
Yhat_list = []
for i in range(img_plt_1.shape[0]):
    for j in range(img_plt_1.shape[1]):
        if 0<img_plt_1[i][j][0]<255 or 0<img_plt_1[i][j][1]<255 or 0<img_plt_1[i][j][2]<255:
            Y_list.append(1)
        else:
            Y_list.append(0)

for i in range(img_plt_2.shape[0]):
    for j in range(img_plt_2.shape[1]):
        if 0<img_plt_1[i][j][0]<255 or 0<img_plt_1[i][j][1]<255 or 0<img_plt_1[i][j][2]<255:
            Yhat_list.append(1)
        else:
            Yhat_list.append(0)
Y=np.array(Y_list)
Yhat=np.array(Yhat_list)
np.savez("data.npz", Y=Y, Yhat=Yhat)
