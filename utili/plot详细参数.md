# 修改字体  
'''
import matplotlib as mpl  
mpl.rcParams['font.sans-serif'] = ['SimHei']  
'''
# 控制颜色 （color）  
plot画图时可以设定线条参数。包括：颜色、线型、标记风格。  

## 颜色之间的对应关系为  
b---blue   c---cyan  g---green    k----black  
m---magenta r---red  w---white    y----yellow  
    =============    ===============================  
    character        color  
    =============    ===============================  
    ``'b'``          blue 蓝  
    ``'g'``          green 绿  
    ``'r'``          red 红  
    ``'c'``          cyan 蓝绿  
    ``'m'``          magenta 洋红  
    ``'y'``          yellow 黄  
    ``'k'``          black 黑  
    ``'w'``          white 白  
    =============    ===============================  
# 控制线型 （linestyle）

符号和线型之间的对应关系

    =============    ===============================
    character        description
    =============    ===============================
    '-'              solid line style 实线
    '--'             dashed line style 虚线
    '-.'             dash-dot line style 点画线
    ':'              dotted line style 点线
    =============    ===============================

# 控制标记风格 （marker）
## 标记风格有多种：
=============    ===============================  
    character           description  
    =============    ===============================  
    ``'.'``          point marker  
    ``','``          pixel marker  
    ``'o'``          circle marker  
    ``'v'``          triangle_down marker  
    ``'^'``          triangle_up marker  
    ``'<'``          triangle_left marker  
    ``'>'``          triangle_right marker  
    ``'1'``          tri_down marker  
    ``'2'``          tri_up marker  
    ``'3'``          tri_left marker  
    ``'4'``          tri_right marker  
    ``'s'``          square marker  
    ``'p'``          pentagon marker  
    ``'*'``          star marker  
    ``'h'``          hexagon1 marker  
    ``'H'``          hexagon2 marker  
    ``'+'``          plus marker  
    ``'x'``          x marker  
    ``'D'``          diamond marker  
    ``'d'``          thin_diamond marker  
    ``'|'``          vline marker  
    ``'_'``          hline marker  
    =============    ===============================  
# plot函数的一般的调用形式：  
plot(x, y)  
    可选参数：  
        color='green'       颜色  
        marker='o'          标记类型  
        markersize=5        标记尺寸  
        markevery=5         标记间隔，比如每5个点做一个标记  
        linestyle='dashed'  线段类型  
        linewidth=1         线段宽度  
# 画子图
import  matplotlib.pyplot as plt  
plt.figure(figsize=(6,6), dpi=80)    # figsize表示画板的大小，dpi为图形的分辨率  
plt.figure(1)  # 表示取第一块画板，一个画板即一张图  
ax1 = plt.subplot(221)   # 221表示将画板分成两行两列，取第一个区域，即左上角区域  
plt.plot([1,2,3,4],[4,5,7,8], color="r",linestyle = "--")  
ax2 = plt.subplot(222)  
plt.plot([1,2,3,5],[2,3,5,7],color="y",linestyle = "-")  
ax3 = plt.subplot(223)  
plt.plot([1,2,3,4],[11,22,33,44],color="g",linestyle = "-.")  
ax4 = plt.subplot(224)  
plt.plot([1,2,3,4],[11,22,33,44],color="b",linestyle = ":")  
