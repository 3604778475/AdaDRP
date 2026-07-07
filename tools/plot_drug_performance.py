import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mc
from matplotlib.patches import Patch

# =========================
# STYLE
# =========================
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

# =========================
# METHODS (9 methods)
# =========================
methods = [
    "Celligner", "Velodrome", "scDEAL",
    "CODE-AE", "WISER", "TransDRP",
    "VAEN", "MACB-DRP", "FMPA"
]
x = np.arange(len(methods))

# =========================
# DRUGS (9 panels)
# =========================
drugs = [
    "5-FU", "Cisplatin", "Cyclophosphamide",
    "Docetaxel", "Doxorubicin", "Etoposide",
    "Gemcitabine", "Paclitaxel", "Temozolomide"
]

# =========================
# COLORS
# =========================
base_colors = [
    "#4c9a2a", "#7f7f7f", "#2f6fdf",
    "#d95f5f", "#f39c34", "#f1c40f",
    "#8e44ad", "#16a085", "#34495e"
]

def lighten(color, factor=0.55):
    c = np.array(mc.to_rgb(color))
    return tuple(1 - (1 - c) * factor)

auc_colors = base_colors
aupr_colors = [lighten(c, 0.55) for c in base_colors]

def text_color(rgb):
    r, g, b = mc.to_rgb(rgb)
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return "black" if lum > 0.6 else "white"

# =========================
# DATA: AUC mean/std (9 methods)
# =========================
auc_mean = {
    "5-FU": [0.5476, 0.4765, 0.5474, 0.5484, 0.6194, 0.7091, 0.5049, 0.6366, 0.5936],
    "Cisplatin": [0.6711, 0.3511, 0.6483, 0.5755, 0.6208, 0.6372, 0.5755, 0.6432, 0.6514],
    "Cyclophosphamide": [0.4907, 0.5683, 0.5761, 0.3553, 0.5461, 0.6217, 0.3553, 0.7102, 0.7063],
    "Docetaxel": [0.4011, 0.5883, 0.4599, 0.4013, 0.5300, 0.5854, 0.4013, 0.5945, 0.5220],
    "Doxorubicin": [0.4484, 0.5609, 0.4756, 0.4668, 0.5082, 0.5792, 0.4668, 0.6219, 0.5791],
    "Etoposide": [0.6919, 0.3589, 0.6563, 0.6542, 0.6773, 0.6298, 0.6542, 0.7670, 0.7864],
    "Gemcitabine": [0.5625, 0.4460, 0.5512, 0.5561, 0.5814, 0.5588, 0.5561, 0.5714, 0.5883],
    "Paclitaxel": [0.5025, 0.5265, 0.4940, 0.4758, 0.5458, 0.5813, 0.4758, 0.6000, 0.5744],
    "Temozolomide": [0.2882, 0.6651, 0.3176, 0.3876, 0.6055, 0.6666, 0.3876, 0.6748, 0.6838],
}

auc_std = {
    "5-FU": [0.0236, 0.0225, 0.0402, 0.0238, 0.0103, 0.1124, 0.0858, 0.0622, 0.0630],
    "Cisplatin": [0.0158, 0.0095, 0.0204, 0.0250, 0.0116, 0.0562, 0.0250, 0.0380, 0.0517],
    "Cyclophosphamide": [0.0394, 0.0178, 0.0988, 0.0459, 0.1012, 0.0791, 0.0459, 0.0646, 0.0302],
    "Docetaxel": [0.0167, 0.0304, 0.0223, 0.0219, 0.0032, 0.1103, 0.0219, 0.0708, 0.0503],
    "Doxorubicin": [0.0732, 0.0338, 0.0249, 0.0783, 0.1249, 0.1229, 0.0783, 0.0670, 0.0645],
    "Etoposide": [0.0497, 0.0228, 0.0676, 0.0585, 0.0725, 0.0435, 0.0585, 0.0837, 0.0536],
    "Gemcitabine": [0.0146, 0.0094, 0.0304, 0.0095, 0.0022, 0.0357, 0.0095, 0.0191, 0.0195],
    "Paclitaxel": [0.0185, 0.0374, 0.0662, 0.0517, 0.0106, 0.0652, 0.0517, 0.0305, 0.0328],
    "Temozolomide": [0.0235, 0.0638, 0.0366, 0.0610, 0.0758, 0.0526, 0.0610, 0.0374, 0.0387],
}

# =========================
# DATA: AUPR mean/std (9 methods)
# =========================
aupr_mean = {
    "5-FU": [0.6873, 0.6151, 0.6742, 0.6784, 0.7002, 0.8050, 0.6794, 0.7373, 0.7106],
    "Cisplatin": [0.8198, 0.6342, 0.8040, 0.7945, 0.7837, 0.8005, 0.7945, 0.8276, 0.8124],
    "Cyclophosphamide": [0.8957, 0.9243, 0.9206, 0.8874, 0.9181, 0.9425, 0.8874, 0.9591, 0.9419],
    "Docetaxel": [0.6387, 0.7466, 0.6620, 0.6288, 0.6664, 0.7478, 0.6288, 0.7563, 0.7148],
    "Doxorubicin": [0.7128, 0.7864, 0.7384, 0.7713, 0.7411, 0.7932, 0.7713, 0.8507, 0.8019],
    "Etoposide": [0.9165, 0.7693, 0.8906, 0.8897, 0.9169, 0.8657, 0.8897, 0.9311, 0.9329],
    "Gemcitabine": [0.5419, 0.4536, 0.5625, 0.5632, 0.5680, 0.5415, 0.5632, 0.5737, 0.5916],
    "Paclitaxel": [0.6705, 0.7109, 0.6768, 0.7061, 0.7142, 0.7782, 0.7061, 0.7715, 0.7491],
    "Temozolomide": [0.1990, 0.4650, 0.2067, 0.2405, 0.3869, 0.4852, 0.2405, 0.5191, 0.5136],
}

aupr_std = {
    "5-FU": [0.0102, 0.0322, 0.0173, 0.0170, 0.0119, 0.0743, 0.1343, 0.0502, 0.0442],
    "Cisplatin": [0.0090, 0.0069, 0.0190, 0.0174, 0.0117, 0.0400, 0.0174, 0.0259, 0.0418],
    "Cyclophosphamide": [0.0161, 0.0037, 0.0299, 0.0174, 0.0288, 0.0134, 0.0174, 0.0153, 0.0146],
    "Docetaxel": [0.0136, 0.0318, 0.0134, 0.0113, 0.0028, 0.0682, 0.0113, 0.0457, 0.0452],
    "Doxorubicin": [0.0335, 0.0244, 0.0153, 0.0495, 0.0626, 0.0845, 0.0495, 0.0270, 0.0495],
    "Etoposide": [0.0079, 0.0055, 0.0262, 0.0217, 0.0197, 0.0085, 0.0217, 0.0342, 0.0204],
    "Gemcitabine": [0.0264, 0.0058, 0.0283, 0.0117, 0.0124, 0.0302, 0.0117, 0.0263, 0.0244],
    "Paclitaxel": [0.0117, 0.0263, 0.0487, 0.0403, 0.0205, 0.0472, 0.0403, 0.0167, 0.0235],
    "Temozolomide": [0.0089, 0.0918, 0.0223, 0.0355, 0.0924, 0.1099, 0.0355, 0.0437, 0.0614],
}

# =========================
# PLOT - 柱状图 (9 methods)
# =========================
fig, axes = plt.subplots(3, 3, figsize=(20, 16), dpi=200)

bar_w = 0.35

for i, drug in enumerate(drugs):
    ax = axes[i // 3, i % 3]

    auc = np.array(auc_mean[drug])
    aupr = np.array(aupr_mean[drug])
    auc_err = np.array(auc_std[drug])
    aupr_err = np.array(aupr_std[drug])

    # AUC bars (深色)
    bars1 = ax.bar(x - bar_w / 2, auc, width=bar_w, yerr=auc_err,
                   color=auc_colors, edgecolor="black", linewidth=0.4,
                   capsize=3, error_kw={'elinewidth': 1.2, 'capthick': 1.2})

    # AUPR bars (浅色)
    bars2 = ax.bar(x + bar_w / 2, aupr, width=bar_w, yerr=aupr_err,
                   color=aupr_colors, edgecolor="black", linewidth=0.4,
                   capsize=3, error_kw={'elinewidth': 1.2, 'capthick': 1.2})

    # =========================
    # VALUE LABEL - 针对Gemcitabine特殊处理
    # =========================
    if drug == "Gemcitabine":
        # Gemcitabine: AUC标签放在柱子下方（偏下），AUPR标签放在柱子上方（偏上）
        # AUC数值标签 - 放在柱子下半部分
        for b, v, c in zip(bars1, auc, auc_colors):
            y_pos = v * 0.25  # 柱子1/4高度（偏下）
            ax.text(b.get_x() + b.get_width() / 2, y_pos,
                    f"{v:.4f}",
                    ha="center", va="center",
                    fontsize=8,
                    fontstyle='italic',
                    color='black',
                    bbox=dict(boxstyle="round,pad=0.15",
                              facecolor="white", alpha=0.20, edgecolor="none"))

        # AUPR数值标签 - 放在柱子上半部分
        for b, v, c in zip(bars2, aupr, aupr_colors):
            y_pos = v * 0.75  # 柱子3/4高度（偏上）
            ax.text(b.get_x() + b.get_width() / 2, y_pos,
                    f"{v:.4f}",
                    ha="center", va="center",
                    fontsize=8,
                    fontstyle='italic',
                    color='black',
                    bbox=dict(boxstyle="round,pad=0.15",
                              facecolor="white", alpha=0.20, edgecolor="none"))
    else:
        # 其他药物：保持原来的位置（都在柱子中间）
        # AUC数值标签 - 统一黑色
        for b, v, c in zip(bars1, auc, auc_colors):
            y_pos = v * 0.5
            ax.text(b.get_x() + b.get_width() / 2, y_pos,
                    f"{v:.4f}",
                    ha="center", va="center",
                    fontsize=8,
                    fontstyle='italic',
                    color='black',
                    bbox=dict(boxstyle="round,pad=0.15",
                              facecolor="white", alpha=0.20, edgecolor="none"))

        # AUPR数值标签 - 统一黑色
        for b, v, c in zip(bars2, aupr, aupr_colors):
            y_pos = v * 0.5
            ax.text(b.get_x() + b.get_width() / 2, y_pos,
                    f"{v:.4f}",
                    ha="center", va="center",
                    fontsize=8,
                    fontstyle='italic',
                    color='black',
                    bbox=dict(boxstyle="round,pad=0.15",
                              facecolor="white", alpha=0.20, edgecolor="none"))

    # 药物标题字体增大加粗
    ax.set_title(drug, fontsize=18, fontweight='bold')
    ax.set_xticks(x)

    # x轴标签按AUC颜色标注，字体加大加粗
    for tick_label, color in zip(ax.get_xticklabels(), auc_colors):
        tick_label.set_color(color)
        tick_label.set_fontweight('bold')

    ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=11, fontweight='bold')

    # 纵轴范围
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(np.arange(0.0, 1.1, 0.1))
    ax.tick_params(axis='y', labelsize=11)

    ax.grid(axis="y", linestyle="--", alpha=0.3)

# =========================
# LEGEND - AUC/AUPR图例
# =========================
legend_elements = [
    Patch(facecolor=auc_colors[0], edgecolor="black", label="AUC"),
    Patch(facecolor=aupr_colors[0], edgecolor="black", label="AUPR")
]

fig.legend(handles=legend_elements,
           loc="lower center",
           ncol=2,
           frameon=False,
           fontsize=14,
           prop={'weight': 'bold'})

plt.subplots_adjust(left=0.05, right=0.98, top=0.94, bottom=0.10,
                    wspace=0.30, hspace=0.50)

plt.savefig("D:/药物反应预测/MACB-DRP-main/outputs/Figure3_BarChart_BoldFont.png", dpi=400, bbox_inches="tight")
plt.show()