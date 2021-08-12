
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: A SHORT DESCRIPTION OF THE PROJECT                                                         -- #
# -- script: visualizations.py : python script with data visualization functions                         -- #
# -- author: YOUR GITHUB USER NAME                                                                       -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: YOUR REPOSITORY URL                                                                     -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""
import matplotlib.pyplot as plt


def basic_plot_x2(df1,df2, title, column1, column2, title_x, title_y, label1, label2):
    plt.figure(figsize=(15,4))
    ax = plt.axes()
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.plot(df1[column1], lw=3,color="green", label=label1)
    plt.plot(df2[column2], lw=3, color="blue", label=label2)
    plt.legend(loc='best', fontsize=8)
    plt.title(title, fontsize=15)
    plt.xlabel(title_x, fontsize=8)
    plt.ylabel(title_y, fontsize=8)
    plt.show()
    return plt.show()


def basic_plot_x4(df1,df2, df3,df4, title, column1, column2, column3,column4, title_x, title_y, label1, label2, label3, label4):
    plt.figure(figsize=(15,4))
    ax = plt.axes()
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.plot(df1[column1], lw=3, color="green", label=label1)
    plt.plot(df2[column2], lw=3, color="blue", label=label2)
    plt.plot(df3[column3], lw=3, color="grey", label=label3)
    plt.plot(df4[column4], lw=3, color="red", label=label4)
    plt.legend(loc='best', fontsize=8)
    plt.title(title, fontsize=15)
    plt.xlabel(title_x, fontsize=8)
    plt.ylabel(title_y, fontsize=8)
    plt.show()
    return plt.show()

