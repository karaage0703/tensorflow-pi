# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("/tmp/tensorflow_pi/train_graph.csv", index_col='step')

fig, ax1 = plt.subplots()
df_accuracy = data.iloc[:, [0]]
df_loss = data.iloc[:, [1]]

ax1.plot(df_accuracy, 'r', label='accuracy')
ax1.set_xlabel('epoch')
ax1.set_ylabel('accuracy')
ax1.set_ylim([0,1])
ax1.legend(loc='upper left')

ax2 = ax1.twinx()
ax2.plot(df_loss, 'b', label='loss')
ax2.set_ylabel('loss')
ax2.set_ylim(bottom=0)
ax2.legend(loc='lower left')

plt.xlim([0,200])
plt.savefig("plot.png")
plt.show()
