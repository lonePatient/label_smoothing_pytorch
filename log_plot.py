import numpy as np
import matplotlib.pyplot as plt
from tools import load_json
plt.switch_backend('agg')  # 防止ssh上绘图问题

data1 = load_json('./logs/ResNet18_training_monitor.json')
data2 = load_json('./logs/ResNet18_label_smoothing_training_monitor.json')

N = np.arange(0, len(data1['loss']))
plt.style.use("ggplot")
plt.figure()
plt.plot(N, data1['loss'], label=f"ResNet18")
plt.plot(N, data2['loss'], label=f"ResNet18_label_smooth")
plt.legend()
plt.xlabel("Epoch #")
plt.ylabel('loss')
plt.title(f"Training loss [Epoch {len(data1['loss'])}]")
plt.savefig('./png/training_loss.png')
plt.close()

N = np.arange(0, len(data1['loss']))
plt.style.use("ggplot")
plt.figure()
plt.plot(N, data1['valid_acc'], label=f"ResNet18")
plt.plot(N, data2['valid_acc'], label=f"ResNet18_label_smooth")
plt.legend()
plt.xlabel("Epoch #")
plt.ylabel('accuracy')
plt.title(f"Valid accuracy [Epoch {len(data1['loss'])}]")
plt.savefig('./png/valid_accuracy.png')
plt.close()