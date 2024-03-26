### 这是一个画出太阳的临边昏暗(Limb Darkening)曲线、帮助找出太阳极区/赤道的代码。version 2.3
# 中山大学物理与天文学院  黄日 (GitHub: betelgeuse-nebula)
# 2023年11月23日

### 确定中心位置与半径的基本原理
# 有规律地或随机取一些像素点，若亮度大于平均值的 jud 倍（Part 1 Line 64，此参数的参考值为 1~5，可根据图像特征调整），则收集；收集的点的位置取平均，作为中心位置的估计值。
# 有规律地取一些像素点（等距取样呗，若有足够算力，那就所有点都取），若亮度大于平均值 jud 倍，则收集；收集的点的个数乘以取样密度，以此估算太阳盘面的面积，再计算半径。

### 本代码处理的图像需要满足以下条件
# fit格式（其他格式我不知道行不行）
# 图像中包含完整的太阳
# 太阳几乎不被云层等物体遮挡
# 可能适用于满月（未尝试），但一定不适用于不满的月

### 本代码的局限性
# 将太阳视作完美的圆处理，没有考虑到太阳扁了
# 不具备将低亮度的黑子（或凌日物体）判定为“盘面内”的能力
# 本代码 Part 7-8 的作用是，画出 mu^2 在很窄的范围内时，能流（信号值）与幅角的关系，可同时画出几个窄带，但这些窄带在本代码只能首尾相连且等距
# ......

### 关于 mu
# 为方便，本代码全程适用 mu^2 处理数据，mu^2 具有以下性质
# 一点到太阳盘面中心的距离（单位：像素）与 mu^2 的值是线性关系：
# 太阳盘面中心 mu^2 = 1
# 太阳盘面之内 mu^2 ∈ (0,1]
# 太阳盘面边缘 mu^2 = 0
# 太阳盘面之外 mu^2 < 0

### 一些乱来的东西（均可设置跳过这些乱来的步骤）
# 本代码确定中心位置的采样点，包含一部分有规律的与一部分随机的点；确定半径的采样点，只包含等距采样（正方形）的点。（若要使采点不随机，即几乎均匀，那么间距最好 <10。若要使中心位置只由随机的点确定（亲测，采点10^7，中心位置仍不太准，不建议用此方法），请注释 Part 1 中 Line 74-75，并最好将 Part 3 中 Line 91 的“采点个数”调整至至少10^6，否则中心位置误差可能会很大）
# 本代码处理后的数据均扣除了天光背景，而天光背景居然采用了均匀的profile。（若不希望乱扣除天光背景，请注释整个 Part 5 与 Part 7 中 Line 217 与 Part 8 中 Line 261）

### 关于坐标  x为横轴，以右为正；y为纵轴，以下为正；Arg以右为0，顺时针为正。

### 【Part 0】准备工作
import numpy as np
from astropy.io import fits
import matplotlib
import matplotlib.pyplot as plt
import random
from scipy.optimize import curve_fit

### 打开文件，导入数据
hdulist = fits.open('example/Preview_20231110_154133_0.05sec_Bin1_43.0C_gain0.fit')
# 将 hdulist 中的数据提取到 data 中
hdu0 = hdulist[0]
data = hdu0.data
data = hdulist[0].data
# 检查是否导入了正确的矩阵 读取矩阵大小
row = np.shape(data)[0]
col = np.shape(data)[1]
print("Data Shape  : ",np.shape(data))
# 计算全图信号值的平均值
avg = np.average(data)
print("Average     : ",avg)




### 【Part 1】识别太阳盘面中心
# 采点间隔
sep = 10  # 建议范围 5~40
# 参数：判定“是否属于盘面”的阈值与平均值之比
jud = 1.3  # 建议范围 1.0~4.0（仅作参考，对于不同的波长、天气状况、具体图像，适合的值不相同）

jud = jud * avg
idenx = []
ideny = []
sizeS = 0
for x in range(0,row,sep):
    for y in range(0,col,sep):
        if data[x,y] > jud:
            sizeS += 1
            idenx.append(x)
            ideny.append(y)




### 【Part 2】计算盘面平均半径
ra = np.sqrt(sep**2*sizeS/np.pi)
print("Radius      : ",ra)
cx = np.average(idenx)  # 盘面中心 x 坐标值（大致估计）
cy = np.average(ideny)  # 盘面中心 y 坐标值（大致估计）




### 【Part 3】进一步校准太阳中心（可选）
# 进一步校准的采点个数（可以取0）
step = 0

tempxmin = np.floor(cx-2*ra).astype(int)
tempxmax = np.floor(cx+2*ra).astype(int)
tempymin = np.floor(cy-2*ra).astype(int)
tempymax = np.floor(cy+2*ra).astype(int)
for i in range(step):
    x = random.randint(tempxmin,tempxmax)
    y = random.randint(tempymin,tempymax)
    if data[x,y] > jud:
        idenx.append(x)
        ideny.append(y)
cx = np.average(idenx)  # 盘面中心 x 坐标值
cy = np.average(ideny)  # 盘面中心 y 坐标值
print("Center Posi :  (",cx,",",cy,")")




### 【Part 4】采点（用于绘制临边昏暗曲线）
# mu^2 的取值个数
mu2num = 128  
# 辐角 phi 的取值个数
phinum = 360
# mu^2 下限（也是天光背景 mu^2 下限，即外边界）
mu2startbg = -0.2
# 天光背景 mu^2 上限，即内边界
mu2start = -0.1
# mu^2 上限
mu2end = 1.0

mu_l = np.zeros((2,mu2num))
ltemp = np.zeros((mu2num,phinum))
mu2 = mu2startbg
ra2 = ra**2  # 提早计算半径 ra[像素] 的平方
for i in range(mu2num):
    # 表达式为 mu^2*ra^2 + di^2 = ra^2, di^2 表示距离
    di = np.sqrt(1 - mu2)*ra
    for j in range(phinum):
        phi = j/phinum * 2*np.pi
        x = np.floor(cx+di*np.cos(phi)+0.5).astype(int)
        y = np.floor(cy+di*np.sin(phi)+0.5).astype(int)
        ltemp[i,j] = data[x,y]
    mu_l[0,i] = mu2
    mu_l[1,i] = np.average(ltemp[i,:])
    mu2 += (mu2end-mu2startbg)/mu2num




### 【Part 5】扣除天光背景（可选）
# 将天光背景近似为均一的，天光采样为 mu^2 取 mu2startbg 到 mu2start 的环状区域
start = np.floor((mu2start-mu2startbg)/(mu2end-mu2startbg)*mu2num).astype(int)
bg = np.average(mu_l[1,0:start])
print("Background  : ",bg)
plt.figure()
plt.rcParams.update({'font.size': 10})
plt.plot(np.linspace(0,360,phinum),ltemp[start,:], marker="+",markersize=5,linewidth=0,color='g',label='Inner boundary ($\mu^2=$%g)' % mu2startbg)
plt.plot(np.linspace(0,360,phinum),ltemp[0,:], marker="x",markersize=5,linewidth=0,color='r',label='Outer boundary ($\mu^2=$%g)' % mu2start)
circle = []
for j in range(phinum):
    circle.append(np.average(ltemp[0:start,j]))
# plt.plot(circle[:],'+b',label='77')
plt.title('Flux of Background Sampling Area')
plt.xlabel('Arg [degree]')
plt.ylabel('Flux [a.u.]')
plt.xlim(0,360)  # 横坐标上下限
plt.xticks(np.linspace(0,360,13,endpoint=True))  # 刻度间隔
plt.rcParams.update({'font.size': 8})
plt.legend(loc='upper right', bbox_to_anchor=(1.40,1.0),borderaxespad = 0.)  # 设置图例位置，放在外面
plt.savefig("1BackgroundFlux.pdf",format="pdf",bbox_inches="tight")

ltemp -= bg
mu_l[1,:] -= bg
# 注：原始数据 data 并没有扣除天光背景




### 【Part 6】画出临边昏暗曲线
plt.figure()
plt.rcParams.update({'font.size': 10})
plt.plot(mu_l[0,:],mu_l[1,:])
plt.title('Limb Darkening Curve of the Sun')
plt.xlabel('$\mu^2$')
plt.ylabel('Flux (background removed) [a.u.]')
plt.xlim(mu2startbg,mu2end)
plt.ylim(0,)
plt.savefig("2LimbDarkeningCurve.pdf",format="pdf",bbox_inches="tight")




### 【Part 7】识别极区的幅角，并画出 mu 在几个很窄的范围内，亮度与幅角的关系
# mu^2 下限（最好>0，即全部在太阳内部）
mu2start = 0.02
# mu^2 上限
mu2end = 0.40
# mu^2 的取值个数
mu2num = 50
# 切割组数
groupnum = 8
# 辐角 phi 的取值个数
phinum = 360

plt.figure()
plt.rcParams.update({'font.size': 10})
for g in range(groupnum-1,-1,-1):
    start = np.floor(mu2num* g   /groupnum).astype(int) #  包含
    ends  = np.floor(mu2num*(g+1)/groupnum).astype(int) #  不包含
    # 组内的离散化的（实际的） mu^2 范围
    mu2lower = mu2start + (mu2end-mu2start)/mu2num * start
    mu2upper = mu2start + (mu2end-mu2start)/mu2num * ends
    ltemp = np.zeros((mu2num,phinum))
    mu2 = mu2lower
    for i in range(start,ends,1):
        # 表达式为 mu^2*ra^2 + di^2 = ra^2, di^2 表示距离
        di = np.sqrt(1 - mu2)*ra
        for j in range(phinum):
            phi = j/phinum * 2*np.pi
            x = np.floor(cx+di*np.cos(phi)+0.5).astype(int)
            y = np.floor(cy+di*np.sin(phi)+0.5).astype(int)
            ltemp[i,j] = data[x,y]
        mu2 += (mu2end-mu2start)/mu2num

    # 扣除背景（可选）
    ltemp -= bg

    circle = []
    for j in range(phinum):
        circle.append(np.average(ltemp[start:ends,j]))
    gg = g+1  # 第 gg 组，从 0 开始不太妙，还是从 1 开始吧
    plt.plot(np.linspace(0,360,phinum),circle[:],label='Ring %g, $\mu^2\in$[%.3f,%.3f)'%(gg,mu2lower,mu2upper), marker="+",markersize=5,linewidth=0)

plt.title('Flux of Edge of Solar Disk')
plt.xlabel('Arg [degree]')
plt.ylabel('Flux (background removed) [a.u.]')
plt.xlim(0,360)  # 横坐标上下限
plt.xticks(np.linspace(0,360,13,endpoint=True))  # 刻度间隔
plt.rcParams.update({'font.size': 8})
plt.legend(loc='upper right', bbox_to_anchor=(1.40,1.0),borderaxespad = 0.)  # 设置图例位置，放在外面
plt.savefig("3SolarDiskFlux.pdf",format="pdf",bbox_inches="tight")




### 【Part 8】画出 mu 在一个很窄的范围内，亮度与幅角的关系，并拟合cos函数
mu2start = 0.02
# mu^2 上限
mu2end = 0.7
# mu^2 的取值个数
mu2num = 50
# 辐角 phi 的取值个数
phinum = 360

plt.figure()
plt.rcParams.update({'font.size': 10})
ltemp = np.zeros((mu2num,phinum))
mu2 = mu2start
for i in range(mu2num):
    # 表达式为 mu^2*ra^2 + di^2 = ra^2, di^2 表示距离
    di = np.sqrt(1 - mu2)*ra
    for j in range(phinum):
        phi = j/phinum * 2*np.pi
        x = np.floor(cx+di*np.cos(phi)+0.5).astype(int)
        y = np.floor(cy+di*np.sin(phi)+0.5).astype(int)
        ltemp[i,j] = data[x,y]
    mu2 += (mu2end-mu2start)/mu2num

# 扣除背景（可选）
# ltemp -= bg

circle = []
for j in range(phinum):
    circle.append(np.average(ltemp[start:ends,j]))
plt.plot(np.linspace(0,360,phinum),circle[:],label='Ring : $\mu^2\in$[%.3f,%.3f)'%(mu2start,mu2end), marker="+",markersize=5,linewidth=0)

def polarcos(x, a, b, c):
    return a * np.cos((x-b)/90*np.pi) + c
popt, pcov = curve_fit(polarcos, np.linspace(0,360,phinum), circle[:], bounds=(0, [5000, 180, 50000]))

# print(popt,pcov)
plt.plot(np.linspace(0,360,phinum), polarcos(np.linspace(0,360,phinum), *popt), 'g--', label='Fit: $A=\cos[2(x-\phi_0)] + C ,\quad A=$%6.2f$\pm$%5.2f, $\phi_0=$%6.2f$\pm$%5.2f, $C=$%6.2f$\pm$%5.2f' % tuple([popt[0],np.sqrt(pcov[0,0]),popt[1],np.sqrt(pcov[1,1]),popt[2],np.sqrt(pcov[2,2])]))

plt.title('Flux of Edge of Solar Disk')
plt.xlabel('Arg [degree]')
plt.ylabel('Flux (background removed) [a.u.]')
plt.xlim(0,360)  # 横坐标上下限
plt.xticks(np.linspace(0,360,13,endpoint=True))  # 刻度间隔
plt.rcParams.update({'font.size': 8})
plt.legend(loc='lower left', bbox_to_anchor=(-0.08,-0.27),borderaxespad = 0.)  # 设置图例位置，放在外面
plt.savefig("4FitCos.pdf",format="pdf",bbox_inches="tight")




### 【Part 9】画出识别的图样
tempxmin = np.floor(cx-1.5*ra).astype(int)
tempxmax = np.floor(cx+1.5*ra).astype(int)
tempymin = np.floor(cy-1.5*ra).astype(int)
tempymax = np.floor(cy+1.5*ra).astype(int)

plt.figure()
plt.imshow(data[tempxmin:tempxmax , tempymin:tempymax], cmap='gray')
dashn = 4
for i in range(dashn):
    dashmin = 2*np.pi * (i + 1/4)/dashn
    dashmax = 2*np.pi * (i + 3/4)/dashn
    plt.plot(cx-tempxmin+ra*np.sin(np.linspace(dashmin,dashmax,360)),cy-tempymin+ra*np.cos(np.linspace(dashmin,dashmax,360)),'r--')
    
plt.plot(cx-tempxmin+1.2*ra*np.cos(np.linspace(popt[1]*np.pi/180,popt[1]*np.pi/180+np.pi,2,endpoint=True)),cy-tempymin+1.2*ra*np.sin(np.linspace(popt[1]*np.pi/180,popt[1]*np.pi/180+np.pi,2,endpoint=True)),'g--')
plt.plot(cx-tempxmin,cy-tempymin, marker="+",markersize=9,linewidth=0,color='red')
plt.savefig("5ShowSun.pdf",format="pdf",bbox_inches="tight")




### 【Part 10】原图
plt.figure()
plt.imshow(data, cmap='gray')
plt.savefig("6Original.pdf",format="pdf",bbox_inches="tight")