# # To test the success rates of 2 functions with the help of COFSelNet
from defunction import *
from bgf import *
from COFSelNetnet import MyModel
L = 10
count = 0
N = 20
d = 2
c = 3.2
centre1=0  #Tumor location
centre2=0
xmin,xmax=-4,4
xmin1,xmax1=xmin+centre1,xmax+centre1
xmin2, xmax2 = xmin+centre2,xmax+centre2
vmax, vmin = 1, -1
centre=torch.tensor([centre1,centre2])   #Tumor location
k = 20
G = 100
w_ini = 0.9
w_end = 0.4
n = 70
number = 1000
COFSelNet = MyModel()
COFSelNet.load_state_dict(torch.load('COFSelNet_weight.pth'))
# def g1(x): return f1(x-centre) / 64.6231 * 1.5 + torch.randn(1) * 0.5
def g1(x): return f2(x-centre) / 12.5399 * 1.5 + torch.randn(1) * 0.5
def g2(x): return f3(x-centre) / 0.9500 * 1.5 + torch.randn(1) * 0.5
# def g2(x): return f4(x-centre) / 32.00 * 1.5 + torch.randn(1) * 0.5
# def g1(x): return f14(x-centre) / 22.2015 * 1.5 + torch.randn(1) * 0.5
# def g1(x): return f15(x-centre) / 103.0457 * 1.5 + torch.randn(1) * 0.5
# def g1(x): return f16(x-centre) / 6.3615 * 1.5 + torch.randn(1) * 0.5
# def g2(x): return f21(x-centre) / 1.2800 * 1.5 + torch.randn(1) * 0.5
# def g2(x): return f22(x-centre) / 60.6375 * 1.5 + torch.randn(1) * 0.5

for oo in range(L):
    o = torch.zeros(3, n, n)
    # x = torch.cat((torch.rand(N, 1) * (xmax1 - xmin1) + xmin1, torch.rand(N, 1) * (xmax2 - xmin2) + xmin2), dim=1)
    x = torch.rand(N, d) * (3 - 1) + -4
    v = torch.rand(N, d) * (vmax - vmin) + vmin
    fitness1 = torch.zeros(N)
    fitness2 = torch.zeros(N)
    flag = 0

    for g in range(G):
        for i in range(N):
            fitness1[i] = g1(x[i, :])
            fitness2[i] = g2(x[i, :])
            fitness1[i], fitness2[i], o = renewl3(
                x[i, 0], x[i, 1], fitness1[i], fitness2[i], n, o,xmin1,xmin2,xmax1,xmax2)

            if torch.norm(x[i, :]-centre) <= 0.1:
                flag = 1
                break

        if g > 1 and flag == 1:
            count += 1
            break

        w = (w_ini - w_end) * (G - g) / G + w_end
        k = 20 - int(g * 10 / G)

        theta1, weak1 = findtheta(fitness1, k, x, N)
        vweak1 = w * v[weak1, :] + c * torch.rand(1) * (theta1 - x[weak1, :])
        v1 = vweak1.expand(N, -1)
        v1 = corrxv1(v1, vmin, vmax)
        x1 = x + v1

        theta2, weak2 = findtheta(fitness2, k, x, N)
        vweak2 = w * v[weak2, :] + c * torch.rand(1) * (theta2 - x[weak2, :])
        v2 = vweak2.expand(N, -1)
        v2 = corrxv1(v2, vmin, vmax)
        x2 = x + v2

        # if torch.rand(1) > 0.5:
        #     v = v1
        #     x = x1
        # else:
        #     v = v2
        #     x = x2

        with torch.no_grad():
            samplelist = ptol(o, n,xmin1,xmin2,xmax1,xmax2)
            samplelist = samplelist.unsqueeze(0)
            x1 = x1.unsqueeze(0)
            x2 = x2.unsqueeze(0)
            out1 = COFSelNet(samplelist, x1)
            out2 = COFSelNet(samplelist, x2)
            if out1 > out2:
                x1 = x1.squeeze()
                v = v1
                x = x1
            else:
                x2 = x2.squeeze()
                v = v2
                x = x2



print("Success rate:", count/L)
